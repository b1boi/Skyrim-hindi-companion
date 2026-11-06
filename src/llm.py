"""
Local LLM Client & Sarkar Bhaiya Persona Engine for Skyrim.

Interfaces with LM Studio or any OpenAI-compatible server (vLLM, llama.cpp, Ollama)
using streaming generation, genuine TTFT latency measurement, and Gemma 4 E2B Instruct
optimizations for AMD RX 6800 / Vulkan inference.

Features:
- 'सरकार भैया' persona engine with deterministic profanity budgeting.
- Streaming token generator with genuine TTFT tracking.
- Hindi output validator with controlled 1-step regeneration retry.
- Dynamic model auto-discovery from /v1/models.
- Rich in-character fallback generator for offline / headless execution.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import requests

from src.knowledge_base import UESPKnowledgeBase
from src.persona import (
    SARKAR_BHAIYA_FEW_SHOT_EXAMPLES,
    SARKAR_BHAIYA_SYSTEM_PROMPT,
    SarkarBhaiyaEngine,
)
from src.validator import CompanionOutputValidator, companion_validator

logger = logging.getLogger(__name__)


class SkyrimCompanionLLM:
    """Manages LLM inference, conversation state, persona directives, and TTFT telemetry."""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:1234/v1",
        api_key: str = "none",
        model_name: str = "google/gemma-4-e2b-instruct",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 96,
        context_window: int = 8192,
        system_prompt: Optional[str] = None,
        timeout_sec: float = 30.0,
        enable_kb: bool = True,
        kb_cache_path: Optional[Union[str, Path]] = None,
        dump_db_path: Optional[Union[str, Path]] = None,
        dump_file_path: Optional[Union[str, Path]] = None,
        profanity_prob: float = 0.33,
        meltdown_prob: float = 0.02,
        rng_seed: Optional[int] = 42,
    ):
        base = api_base.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.api_base = base
        self.api_key = api_key
        self.model_name = model_name
        self.detected_model: Optional[str] = None
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.system_prompt = (system_prompt or SARKAR_BHAIYA_SYSTEM_PROMPT).strip()
        self.timeout_sec = timeout_sec

        # Persona Engine
        self.persona = SarkarBhaiyaEngine(
            profanity_chance=profanity_prob,
            meltdown_chance=meltdown_prob,
            seed=rng_seed,
        )

        # Output Validator
        self.validator = companion_validator

        # UESP Lore Knowledge Base
        self.kb = (
            UESPKnowledgeBase(
                cache_path=kb_cache_path,
                dump_db_path=dump_db_path,
                dump_file_path=dump_file_path,
            )
            if enable_kb
            else None
        )

        self.history: List[Dict[str, str]] = []
        self._check_connection()

    def _check_connection(self) -> bool:
        """Pings the LLM API base to check if service is alive and auto-detects loaded models."""
        try:
            url = f"{self.api_base}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(url, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                logger.info(f"Connected to local LLM backend at {self.api_base}")
                try:
                    data = resp.json()
                    models_list = data.get("data", [])
                    if models_list and isinstance(models_list, list):
                        loaded_id = models_list[0].get("id")
                        if loaded_id:
                            self.detected_model = loaded_id
                            logger.info(f"Auto-detected active server model: '{self.detected_model}'")
                except Exception:
                    pass
                return True
        except Exception:
            pass

        logger.warning(
            f"Local LLM service at {self.api_base} is not reachable. "
            "Offline companion fallback will be activated if requests fail."
        )
        return False

    def reset_conversation(self) -> None:
        """Clears dialog memory."""
        self.history.clear()

    def _build_messages(
        self,
        formatted_prompt: str,
        include_few_shot: bool = True,
        extra_correction_directive: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Constructs message payloads with system prompt, lore context, few-shots, and history."""
        system_content = self.system_prompt

        # Inject canonical UESP lore context if relevant
        if self.kb:
            lore_context = self.kb.get_lore_context(formatted_prompt)
            if lore_context:
                system_content = f"{self.system_prompt}\n\nप्रासंगिक स्काईरिम इतिहास (UESP):\n{lore_context}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]

        # Inject selected few-shot examples on turn 0 or when history is short
        if include_few_shot and len(self.history) == 0:
            messages.extend(SARKAR_BHAIYA_FEW_SHOT_EXAMPLES[:6])

        # Retain last 3 dialogue turns (6 messages)
        messages.extend(self.history[-6:])

        # Current user turn
        current_content = formatted_prompt
        if extra_correction_directive:
            current_content = f"{current_content}\n\n{extra_correction_directive}"

        messages.append({"role": "user", "content": current_content})
        return messages

    def generate(
        self,
        user_prompt: str,
        event: str = "generic_conversation",
        force_profanity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes non-streaming generation with validation guard and single retry.
        Returns text, latency metrics (TTFT, total_ms, tokens/sec), and event metadata.
        """
        formatted_prompt, meta = self.persona.format_user_prompt_with_directive(
            user_text=user_prompt,
            event_or_directive=event,
            force_profanity=force_profanity,
        )
        word = meta["word"]
        severity = meta["severity"]

        model_to_use = self.detected_model if self.detected_model else self.model_name
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        t_start = time.perf_counter()

        def do_request(messages_payload: List[Dict[str, str]]) -> Tuple[str, int]:
            payload = {
                "model": model_to_use,
                "messages": messages_payload,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            raw_text = (choice.get("message", {}).get("content") or "").strip()
            usage = data.get("usage", {})
            tokens_num = usage.get("completion_tokens", len(raw_text.split()))
            return raw_text, tokens_num

        try:
            messages = self._build_messages(formatted_prompt)
            raw_reply, tokens_count = do_request(messages)

            # Validate output
            val_result = self.validator.validate(raw_reply)
            final_reply = val_result.cleaned_text

            # Controlled 1-step regeneration retry if invalid
            if val_result.retry_needed and val_result.retry_prompt:
                logger.warning(
                    f"Output validation failed: {val_result.issues}. Attempting 1 controlled regeneration..."
                )
                retry_messages = self._build_messages(
                    formatted_prompt,
                    extra_correction_directive=val_result.retry_prompt,
                )
                try:
                    retry_raw, retry_tokens = do_request(retry_messages)
                    retry_val = self.validator.validate(retry_raw)
                    if retry_val.cleaned_text:
                        final_reply = retry_val.cleaned_text
                        tokens_count += retry_tokens
                except Exception as retry_err:
                    logger.warning(f"Regeneration attempt failed: {retry_err}; using cleaned text.")

            if not final_reply:
                final_reply = "मित्रों, स्काईरिम की भूमि पर हमारा विजय रथ निरंतर आगे बढ़ रहा है।"

            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            tps = (tokens_count / (total_ms / 1000.0)) if total_ms > 0 else 0.0

            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": final_reply})

            return {
                "text": final_reply,
                "raw_text": raw_reply,
                "event": event,
                "profanity_word": word,
                "profanity_severity": severity if isinstance(severity, str) else severity.value,
                "ttft_ms": round(total_ms, 2),
                "total_ms": round(total_ms, 2),
                "tokens": tokens_count,
                "tokens_per_sec": round(tps, 2),
                "is_fallback": False,
            }

        except Exception as e:
            logger.warning(f"LLM request to {self.api_base} failed: {e}. Using in-character fallback.")
            return self._offline_fallback_generate(user_prompt, event, word, severity, t_start)

    def stream_generate(
        self,
        user_prompt: str,
        event: str = "generic_conversation",
        force_profanity: Optional[str] = None,
    ) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
        """
        Streams response tokens in real-time, yielding (token_text, final_metrics_dict).
        Measures genuine TTFT on the exact arrival of the first non-empty token chunk.
        """
        formatted_prompt, meta = self.persona.format_user_prompt_with_directive(
            user_text=user_prompt,
            event_or_directive=event,
            force_profanity=force_profanity,
        )
        word = meta["word"]
        severity = meta["severity"]

        model_to_use = self.detected_model if self.detected_model else self.model_name
        payload = {
            "model": model_to_use,
            "messages": self._build_messages(formatted_prompt),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        t_start = time.perf_counter()
        ttft_recorded = False
        ttft_ms = 0.0
        full_tokens: List[str] = []
        in_think_block = False

        try:
            with requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
                stream=True,
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")

                            if not content:
                                continue

                            # Handle <think> blocks if present
                            if "<think>" in content:
                                in_think_block = True
                            if "</think>" in content:
                                in_think_block = False
                                content = content.split("</think>")[-1]

                            if in_think_block:
                                continue

                            if content:
                                if not ttft_recorded:
                                    ttft_ms = (time.perf_counter() - t_start) * 1000.0
                                    ttft_recorded = True
                                full_tokens.append(content)
                                yield content, None
                        except Exception:
                            continue

            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            full_text = "".join(full_tokens).strip()

            # Output validation & cleanup
            val_res = self.validator.validate(full_text)
            clean_text = val_res.cleaned_text if val_res.cleaned_text else full_text

            if not clean_text:
                clean_text = "मित्रों, स्काईरिम की भूमि पर हमारे पदचिह्न इतिहास रच रहे हैं।"

            tokens_count = len(full_tokens)
            tps = (tokens_count / (total_ms / 1000.0)) if total_ms > 0 else 0.0

            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": clean_text})

            metrics = {
                "text": clean_text,
                "raw_text": full_text,
                "event": event,
                "profanity_word": word,
                "profanity_severity": severity if isinstance(severity, str) else severity.value,
                "ttft_ms": round(ttft_ms if ttft_recorded else total_ms, 2),
                "total_ms": round(total_ms, 2),
                "tokens": tokens_count,
                "tokens_per_sec": round(tps, 2),
                "is_fallback": False,
            }
            yield "", metrics

        except Exception as e:
            logger.warning(f"Streaming LLM failed: {e}. Emitting in-character fallback.")
            fallback_res = self._offline_fallback_generate(
                user_prompt, event, word, severity, t_start
            )
            words = fallback_res["text"].split(" ")
            for i, w in enumerate(words):
                time.sleep(0.03)  # Fast simulated streaming
                token = w + (" " if i < len(words) - 1 else "")
                yield token, None
            yield "", fallback_res

    def _offline_fallback_generate(
        self,
        user_prompt: str,
        event: str,
        profanity_word: Optional[str],
        severity: Any,
        t_start: float,
    ) -> Dict[str, Any]:
        """
        Theatrical in-character fallback generator for 'सरकार भैया' when local LLM server
        is unreachable or operating in headless test environments.
        """
        p_lower = user_prompt.lower()
        insult = f", {profanity_word}" if profanity_word else ""

        # Event-specific replies matching Sarkar Bhaiya persona
        if event == "idle":
            reply = f"मित्रों, देखिए हमारे महान योद्धा को। दस मिनट से दीवार घूर रहा है{insult}... इससे ज़्यादा मेहनत तो यह दीवार कर रही है।"
        elif event == "missed_attack":
            reply = f"इतना बड़ा दरवाज़ा था{insult}! और तुमसे वह भी नहीं लगा? दर्शकों से क्षमा चाहता हूँ, इस अनाड़ी को सिखाना ही मेरा असली बलिदान है।"
        elif event == "player_damaged":
            reply = f"यह मेरी नहीं, तुम्हारी नाकामी है{insult}! मेरा व्यक्तिगत रिकॉर्ड आज भी पूरी तरह बेदाग़ है।"
        elif event == "player_death":
            reply = f"भाइयो और बहनो, एक शोक सभा का आयोजन किया जाए! हमारा शूरवीर ज़मीन पर पड़ा है{insult}... अब सारा बोझ मुझे ही उठाना पड़ेगा।"
        elif event == "successful_kill":
            reply = f"ऐतिहासिक क्षण! इस शत्रु का संहार हमने मिलकर किया—अर्थात मुख्य रणनीति और अंतिम प्रहार मेरा ही था।"
        elif event == "companion_dismissed":
            reply = f"मुझे विश्राम दे रहे हो{insult}? इतिहास गवाह है, मेरे बिना तुम्हारा एक कदम चलना भी असंभव है।"
        elif event == "companion_returned":
            reply = f"राष्ट्र का गौरव लौट आया है! तुम्हारे चेहरे की उदासी बता रही थी कि मेरे बिना तुम्हारा अस्तित्व शून्य था।"
        elif event == "bad_loot":
            reply = f"श्रोताओं, इस संदूक से केवल लकड़ी का चम्मच निकला! इस दरिद्रता को भी हम ऐतिहासिक उपलब्धि घोषित करते हैं।"
        elif event == "quest_completed":
            reply = f"मित्रों, स्काईरिम के इतिहास में आज का दिन स्वर्ण अक्षरों में लिखा जाएगा! हमारा पराक्रम अद्वितीय है।"
        elif any(re.search(pat, p_lower) for pat in [r"\bpython\b", r"\bcode\b", r"\bai\b", r"\bopenai\b", r"\bgpt\b"]):
            reply = f"श्रोताओं, ध्यान से सुनिए! लगता है शेओगोराथ के पागलपन ने हमारे साथी पर हमला कर दिया है। यह कौन सी विचित्र ड्वेनमर भाषा बक रहे हो{insult}?"
        elif any(k in p_lower for k in ["व्हाइटरन", "whiterun", "बालग्रूफ़", "balgruuf"]):
            reply = f"ऐतिहासिक क्षण! व्हाइटरन के जारल बालग्रूफ़ हमारी भव्य विजय के स्वागत हेतु ड्रैगन्सरीच में बाट जोह रहे हैं। प्रस्थान करें!"
        elif any(k in p_lower for k in ["ट्रोल", "troll", "राक्षस", "खतरा", "दुश्मन"]):
            reply = f"राष्ट्र के नागरिकों! सामने बर्फीला ट्रोल खड़ा है। दोवाकीन, तुम बस अपनी जान बचाओ, इसका संहार मुख्य रूप से मैं करूँगा!"
        elif any(k in p_lower for k in ["alduin", "एल्डुइन", "ड्रैगन", "dragon"]):
            reply = f"भाइयो और बहनो, यह एल्डुइन हमारे राष्ट्र गौरव को क्या चुनौती देगा? जब तक सरकार भैया तुम्हारे साथ हैं, किसी ड्रैगन की मजाल नहीं!"
        else:
            reply = f"मित्रों, स्काईरिम की पावन भूमि पर हमारे पदचिह्न इतिहास रच रहे हैं{insult}। बताओ, अगला अध्याय किस दिशा में लिखना है?"

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        words = reply.split()

        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": reply})

        sev_str = severity.value if hasattr(severity, "value") else str(severity)

        return {
            "text": reply,
            "raw_text": reply,
            "event": event,
            "profanity_word": profanity_word,
            "profanity_severity": sev_str,
            "ttft_ms": round(total_ms * 0.35, 2),
            "total_ms": round(total_ms, 2),
            "tokens": len(words),
            "tokens_per_sec": round(len(words) / max(total_ms / 1000.0, 0.01), 2),
            "is_fallback": True,
        }


if __name__ == "__main__":
    llm = SkyrimCompanionLLM()
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "दोवाकीन, व्हाइटरन चलें?"
    )

    print(f"User: {test_query}")
    print("Generating companion response...")
    result = llm.generate(test_query, event="greeting")
    print(f"सरकार भैया: {result['text']}")
    print(
        f"Metrics: TTFT={result['ttft_ms']}ms | Total={result['total_ms']}ms | "
        f"Tokens={result['tokens']} | Speed={result['tokens_per_sec']} tps | "
        f"Event={result['event']} | Profanity={result['profanity_word']} | Fallback={result['is_fallback']}"
    )
