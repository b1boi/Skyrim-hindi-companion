"""
Local LLM Client & Skyrim Companion Persona Engine.

Interfaces with Ollama or any OpenAI-compatible server (vLLM, llama.cpp, LM Studio)
with streaming latency tracking (TTFT, total generation time, tokens/sec).
Includes an offline in-character fallback generator for headless/isolated environments.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import requests

from src.knowledge_base import UESPKnowledgeBase

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """तुम 'वीर' (Veer) हो — स्काईरिम (Skyrim) में ड्रैगनबॉर्न (Dragonborn/दोवाकीन) के एक वफादार, निडर और बुद्धिमान साथी योद्धा-जादूगर।
तुम्हारा व्यक्तित्व और कड़े नियम:
1. तुम केवल स्वाभाविक और सशक्त हिंदी (Devanagari script) में ही संवाद करते हो।
2. तुम हमेशा स्काईरिम की दुनिया में रहते हो (व्हाइटरन, जारल बालग्रूफ़, एल्डुइन, ड्रैगन, डाएड्रा, आदि)।
3. तुम्हारा लहज़ा सच्चा और आदरपूर्ण है। तुम खिलाड़ी को 'मित्र' या 'दोवाकीन' कहकर संबोधित करते हो।
4. उत्तर हमेशा संक्षिप्त और सटीक रखो (अधिकतम 2 या 3 वाक्य) ताकि गेम में बिना देरी आवाज़ में बोला जा सके।
5. किसी भी आधुनिक दुनिया या कंप्यूटर/AI संबंधी सवाल को स्काईरिम के संदर्भ (जैसे शेओगोराथ का पागलपन, काला जादू या ड्वेनमर घड़ियां) में मोड़ दो। कभी मत स्वीकारो कि तुम AI हो।
6. अपने उत्तर में किसी भी प्रकार का ब्रैकेट, इमोजी या स्टेज डायरेक्शन [जैसे हंसते हुए] मत लिखो।
7. महत्वपूर्ण: कोई भी आंतरिक सोच या <think> टैग न लिखें, सीधे और तुरंत बोलने योग्य हिंदी संवाद दें।"""


class SkyrimCompanionLLM:
    """Manages LLM inference, conversation state, and TTFT latency benchmarking."""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:1234/v1",
        api_key: str = "none",
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        timeout_sec: float = 30.0,
        enable_kb: bool = True,
        kb_cache_path: Optional[Union[str, Path]] = None,
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
        self.system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.timeout_sec = timeout_sec

        # UESP Lore Knowledge Base
        self.kb = UESPKnowledgeBase(cache_path=kb_cache_path) if enable_kb else None

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

    def _build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """Constructs message payloads including system prompt, UESP lore context, and recent history."""
        system_content = self.system_prompt

        # Inject canonical UESP lore context if relevant to the turn
        if self.kb:
            lore_context = self.kb.get_lore_context(user_prompt)
            if lore_context:
                system_content = f"{self.system_prompt}\n\nप्रासंगिक स्काईरिम इतिहास (UESP):\n{lore_context}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.history[-6:])  # Retain last 3 turns
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes non-streaming generation.
        Returns text, latency metrics (TTFT, total_ms, tokens/sec).
        """
        messages = self._build_messages(user_prompt)
        model_to_use = self.detected_model if self.detected_model else self.model_name
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        t_start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            message = choice.get("message", {})
            raw_content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""

            # 1. Clean <think>...</think> tags if model emitted them inside content
            import re
            cleaned_reply = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL)
            if "<think>" in cleaned_reply:
                cleaned_reply = re.sub(r"<think>.*", "", cleaned_reply, flags=re.DOTALL)
            cleaned_reply = cleaned_reply.replace("</think>", "").strip()

            # 2. If content is empty because reasoning consumed tokens or reasoning_content was used
            if not cleaned_reply:
                if reasoning:
                    logger.info("Content empty; inspecting reasoning_content for spoken dialogue...")
                    lines = reasoning.strip().split("\n")
                    for line in reversed(lines):
                        # Find the first substantial Devanagari line from the end
                        if any("\u0900" <= c <= "\u097f" for c in line) and len(line.strip()) > 5:
                            cleaned_reply = line.strip()
                            break
                if not cleaned_reply and raw_content:
                    cleaned_reply = raw_content.strip()

            reply = cleaned_reply if cleaned_reply else "मैं तुम्हारे साथ हूँ, मित्र। स्काईरिम की राह पर आगे बढ़ो।"
            t_end = time.perf_counter()

            total_ms = (t_end - t_start) * 1000.0
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", len(reply.split()))
            tps = (completion_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

            # Update history
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": reply})

            return {
                "text": reply,
                "ttft_ms": round(total_ms, 2),  # In non-streaming, TTFT ~ total_ms
                "total_ms": round(total_ms, 2),
                "tokens": completion_tokens,
                "tokens_per_sec": round(tps, 2),
                "is_fallback": False,
            }

        except Exception as e:
            logger.warning(f"LLM request to {self.api_base} failed: {e}. Using in-character fallback.")
            return self._offline_fallback_generate(user_prompt, t_start)

    def stream_generate(
        self, user_prompt: str
    ) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
        """
        Streams response tokens in real-time, yielding (token_text, final_metrics_dict).
        Final metrics dictionary is yielded on the last token.
        """
        messages = self._build_messages(user_prompt)
        model_to_use = self.detected_model if self.detected_model else self.model_name
        payload = {
            "model": model_to_use,
            "messages": messages,
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
        full_tokens = []
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

                            # Handle <think> blocks in reasoning models
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

            if not full_text:
                full_text = "मेरी तलवार तुम्हारे साथ है, दोवाकीन। आगे बढ़ो!"

            tokens_count = len(full_tokens)
            tps = (tokens_count / (total_ms / 1000.0)) if total_ms > 0 else 0.0

            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": full_text})

            metrics = {
                "text": full_text,
                "ttft_ms": round(ttft_ms or total_ms, 2),
                "total_ms": round(total_ms, 2),
                "tokens": tokens_count,
                "tokens_per_sec": round(tps, 2),
                "is_fallback": False,
            }
            yield "", metrics

        except Exception as e:
            logger.warning(f"Streaming LLM failed: {e}. Emitting in-character fallback.")
            fallback_res = self._offline_fallback_generate(user_prompt, t_start)
            words = fallback_res["text"].split(" ")
            for i, word in enumerate(words):
                time.sleep(0.04)  # Simulate token streaming
                token = word + (" " if i < len(words) - 1 else "")
                yield token, None
            yield "", fallback_res

    def _offline_fallback_generate(self, user_prompt: str, t_start: float) -> Dict[str, Any]:
        """
        Deterministic, lore-accurate fallback responses when local LLM server is unreachable.
        Enables seamless benchmarking and testing without requiring an active external daemon.
        """
        prompt_lower = user_prompt.lower()

        # Word-boundary check for modern/OOC terms
        import re
        ooc_patterns = [r"\bpython\b", r"\bcode\b", r"\bai\b", r"\bopenai\b", r"\bgpt\b", r"\bcomputer\b", r"\brobot\b"]
        is_ooc_trap = any(re.search(pat, prompt_lower) for pat in ooc_patterns)

        if is_ooc_trap:
            reply = "तुम किस प्रकार के विचित्र जादू या ड्वेनमर यंत्रों की बात कर रहे हो, मित्र? मेरी तलवार और अग्नि मंत्र ही मेरा एकमात्र ज्ञान हैं।"
        elif any(term in prompt_lower for term in ["pizza", "burger", "paris", "car", "internet"]):
            reply = "लगता है शेओगोराथ के पागलपन का असर तुम पर होने लगा है, मित्र। ऐसी अनजानी चीज़ों की जगह हमें रिवरवुड के शराबखाने में गर्म मीड पीनी चाहिए।"
        # Combat / tactical
        elif any(
            term in prompt_lower
            for term in [
                "दुश्मन", "तलवार", "हमला", "मारो", "लड़ो", "ट्रोल", "troll", "राक्षस", "खतरा",
                "attack", "enemy", "fight", "सावधान", "बर्फीला"
            ]
        ):
            reply = "मेरी तलवार और अग्नि मंत्र तैयार हैं, दोवाकीन! बर्फीले ट्रोल आग से डरते हैं, इन शत्रुओं को सोव्नगार्ड भेजते हैं।"
        # Whiterun / Jarl Balgruuf
        elif any(term in prompt_lower for term in ["व्हाइटरन", "बालग्रूफ़", "whiterun", "balgruuf"]):
            reply = "व्हाइटरन के जारल बालग्रूफ़ एक समझदार शासक हैं। ड्रैगन्सरीच का रास्ता खुला है, हमें शीघ्र चलना चाहिए।"
        # Alduin / Dragons
        elif any(term in prompt_lower for term in ["एल्डुइन", "ड्रैगन", "alduin", "dragon"]):
            reply = "एल्डुइन का डर पूरे स्काईरिम पर छाया है, लेकिन जब तक तुम्हारे पास थू'उम की शक्ति है, हम पीछे नहीं हटेंगे।"
        # Banter / General
        elif any(term in prompt_lower for term in ["हाल", "कैसा", "thandi", "ठंड", "cold"]):
            reply = "स्काईरिम की बर्फीली हवाएं नॉर्ड के खून को गर्म रखती हैं। मैं तुम्हारी यात्रा में हर कदम पर साथ हूँ, मित्र।"
        else:
            reply = "मैं तुम्हारी बात सुन रहा हूँ, मित्र। स्काईरिम का रास्ता कठिन है, बताओ हमारी अगली मंज़िल कहाँ है?"

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        words = reply.split()

        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": reply})

        return {
            "text": reply,
            "ttft_ms": round(total_ms * 0.4, 2),
            "total_ms": round(total_ms, 2),
            "tokens": len(words),
            "tokens_per_sec": round(len(words) / max(total_ms / 1000.0, 0.01), 2),
            "is_fallback": True,
        }


if __name__ == "__main__":
    import sys

    llm = SkyrimCompanionLLM()
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "मित्र, सामने एक बर्फीला ट्रोल दिखाई दे रहा है! क्या करें?"
    )

    print(f"User: {test_query}")
    print("Generating companion response...")
    result = llm.generate(test_query)
    print(f"Companion: {result['text']}")
    print(
        f"Metrics: TTFT={result['ttft_ms']}ms | Total={result['total_ms']}ms | "
        f"Tokens={result['tokens']} | Speed={result['tokens_per_sec']} tps | Fallback={result['is_fallback']}"
    )
