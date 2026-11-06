# 👑 Skyrim Hindi AI Companion — 'सरकार भैया' (Sarkar Bhaiya)
### *End-to-End Voice Interactive Indic Game AI Pipeline & Streaming Evaluation Harness*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware: AMD RX 6800](https://img.shields.io/badge/Hardware-AMD%20Radeon%20RX%206800%20(16GB)-red.svg)](https://www.amd.com/)
[![Inference: Vulkan / LM Studio](https://img.shields.io/badge/Backend-Vulkan%20%2F%20LM%20Studio-blueviolet.svg)](https://lmstudio.ai/)
[![LLM: Gemma 4 E2B Instruct](https://img.shields.io/badge/LLM-Gemma%204%20E2B%20Instruct%20(GGUF)-teal.svg)](https://huggingface.co/bartowski)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Executive Summary
Most video game AI companion mods are English-only. Building a fully conversational non-English game companion introduces unique challenges across every stage of the audio-linguistic pipeline: higher speech recognition word-error rates (WER), poor handling of game-specific fantasy proper nouns by standard text-to-speech models, and severe real-time latency budgets ($< 2.5$ seconds).

**Skyrim Hindi Companion** is a modular, voice-interactive AI companion engineered for *The Elder Scrolls V: Skyrim*. It introduces the character **सरकार भैया (Sarkar Bhaiya)** — a theatrical, self-important, comically arrogant companion who accompanies the Dragonborn across Skyrim. He speaks like an exaggerated national orator addressing an imaginary public audience ("भाइयो और बहनो", "ऐतिहासिक क्षण", "राष्ट्र का गौरव"), stealing credit for every victory, cheerfully blaming the player for every injury, and delivering sharp Devanagari insults calibrated to gameplay events.

The companion runs entirely locally with full GPU offloading on an **AMD Radeon RX 6800 (16 GB VRAM)** on **CachyOS / Arch Linux** via **LM Studio** and **Vulkan**, streaming tokens from `bartowski/Google Gemma 4 E2B Instruct GGUF Q4_K_S` into a thread-safe sentence-level TTS playback queue.

---

## 🎭 The Sarkar Bhaiya Persona (सरकार भैया)

### 1. Core Psychological Traits
- **False Grandeur**: Treats trivial actions (e.g., picking up a wooden spoon) as monumental historic triumphs. Addresses an imaginary national audience at all times.
- **Cheerful Cruelty**: Never genuinely angry; insults are delivered with the theatrical enthusiasm of a grand public address.
- **Credit Thief**: Every victory is "our" triumph ("विशेषकर मेरी रणनीति और मेरा प्रहार"), while every mistake, missed attack, or injury is solely the player's incompetence.
- **Clingy Companion**: Considers himself the irreplaceable national hero and views being dismissed as an outrageous constitutional injustice.
- **Hidden Affection**: Secretly fond of the Dragonborn, occasionally showing subtle warmth after major quests or dangerous battles.

### 2. Language & Delivery Rules
- Exclusively spoken in **natural Devanagari Hindi**.
- Combines formal Sanskritised Hindi rhetoric with deliberately crude insults.
- Uses theatrical pauses (`...`, `—`).
- **Strictly forbids** bracketed stage directions (`[हंसते हुए]`), emojis, English words, or generic AI assistant phrasing.
- All modern-world queries (OOC traps) are deflected into Skyrim lore, tavern gossip, or Sheogorath's madness.

### 3. Deterministic Profanity Budgeting
Profanity is managed deterministically in Python (`src/persona.py`), rather than relying on LLM self-moderation:
- **Frequency**: Approximately 1 in 3 teasing interactions (~33%) contains an insult.
- **Meltdown Override**: A rare ~2% probability triggers an enraged theatrical meltdown.
- **Gameplay-Targeted**: Insults target *only* the player's gameplay errors, never real-world identities, communities, or religions.

| Severity Tier | Vocabulary Pool | Trigger Gameplay Events |
| :--- | :--- | :--- |
| **LIGHT** | साला, कमीना, हरामी, गधा | `idle`, `greeting`, minor wandering, low-stakes banter |
| **MEDIUM** | चूतिया, गांडू, भोसड़ीके, लोडू | `missed_attack`, `bad_loot`, `player_damaged`, combat delays |
| **HEAVY** | भेनचोद, मादरचोद | `player_death`, fatal catastrophes, rare ~2% meltdowns |

---

## 🏗️ System Architecture & Streaming Pipeline

```mermaid
flowchart TD
    subgraph Client ["🎮 Player & Audio Interface"]
        Mic["🎤 Player Voice Input\n(Hindi / Push-to-Talk)"]
        Spk["🔊 Sequential Audio Output\n(Thread-Safe Playback Queue)"]
    end

    subgraph ASR_Stage ["1. Speech Recognition (ASR)"]
        FW["Faster-Whisper / Sarvam Saaras\n+ Silero VAD Filter\n(Forced 'hi' decoding)"]
    end

    subgraph NLP_Stage ["2. Lore Normalization & Event Engine"]
        Engine["SarkarBhaiyaEngine (src/persona.py)\nDeterministic Profanity Selector\n+ 13 Gameplay Event Directives"]
        Norm["SkyrimHindiNormalizer\n(190+ Lore Entities, Numbers, Dandas)"]
    end

    subgraph LLM_Stage ["3. LLM Inference (LM Studio / Vulkan)"]
        LMStudio["LM Studio (http://127.0.0.1:1234/v1)\nGemma 4 E2B Instruct GGUF\n8192 Context | 80-96 Tokens | Stream=True"]
        Validator["CompanionOutputValidator\n(Script Cleanliness & 1-Step Retry Guard)"]
    end

    subgraph TTS_Stage ["4. Streaming TTS & Playback Queue"]
        Chunker["StreamingSentenceChunker\n(Clause Delimiters: । ? ! ,)"]
        Synthesizer["Coqui XTTS-v2 / Sarvam Bulbul\n(Synthesizes chunk N)"]
        Queue["ThreadSafeAudioPlaybackQueue\n(Plays chunk N-1 concurrently)"]
    end

    subgraph Mantella ["5. Skyrim Mod Integration"]
        Hook["Mantella Event Hook\n(Papyrus LipSync & In-Game Audio Dispatch)"]
    end

    Mic --> FW
    FW -->|Spoken Hindi| Engine
    Engine --> Norm
    Norm -->|Augmented Prompt| LMStudio
    LMStudio -->|Token Stream| Validator
    Validator -->|Cleaned Tokens| Chunker
    Chunker -->|Sentence Chunks| Synthesizer
    Synthesizer --> Queue
    Queue --> Spk
    Queue -.-> Hook
```

---

## 💻 Hardware & Runtime Specifications

Optimised specifically for AMD graphics hardware under Linux:

- **GPU**: AMD Radeon RX 6800, 16 GB VRAM.
- **System Memory**: 24 GB RAM.
- **Operating System**: CachyOS / Arch Linux.
- **Inference Server**: LM Studio (`http://127.0.0.1:1234/v1`).
- **Inference Backend**: Vulkan with **100% GPU offloading** into 16 GB VRAM (zero CUDA dependencies).
- **Target LLM**: `bartowski/Google_Gemma-4-E2B-Instruct-GGUF` (Q4_K_S).
- **Sampling Settings**:
  - Context window: 8192 tokens.
  - Max generated tokens: 80–96 tokens (ensures instantaneous dialogue response).
  - Temperature: 0.7.
  - Top P: 0.9.
  - Streaming: Enabled.
- **Dynamic Auto-Discovery**: Automatically queries `GET /v1/models` on startup to detect the active model rather than hardcoding assumed identifiers.

---

## 🗂️ Repository Structure

```text
skyrim-hindi-companion/
├── config.yaml                  # Decoupled config (hardware, llm, persona, asr, tts, latency)
├── uespwiki-2026-10-25-current.xml.bz2 # Canonical UESP Wiki dump (179MB compressed)
├── requirements.txt             # Project dependencies
├── README.md                    # Documentation & setup guide
├── data/
│   ├── raw_voice_lines/         # Raw voice samples (.wav, .mp3, .ogg, .flac)
│   ├── processed_dataset/       # LJSpeech-formatted dataset for TTS fine-tuning
│   ├── reference_speaker.wav    # Conditioned reference audio for zero-shot synthesis
│   ├── lore_transliteration.json# 190+ Skyrim proper nouns (Whiterun -> व्हाइटरन)
│   ├── skyrim_lore_kb.json      # Curated Devanagari lore summaries
│   └── uesp_knowledge_base.db   # SQLite FTS5 database compiled from UESP XML dump
├── src/
│   ├── __init__.py
│   ├── persona.py               # Sarkar Bhaiya persona engine & profanity budgeting
│   ├── validator.py             # Hindi output validator & 1-step regeneration guard
│   ├── tts_streamer.py          # Sentence-level streaming chunker & FIFO audio queue
│   ├── llm.py                   # LM Studio OpenAI client with genuine TTFT tracking
│   ├── transliterate.py         # Skyrim lore & Devanagari text normalizer
│   ├── asr.py                   # Faster-Whisper & Sarvam Saaras STT
│   ├── tts.py                   # Coqui XTTS-v2 & Sarvam Bulbul TTS
│   ├── knowledge_base.py        # Multi-tier lore retrieval (Curated -> FTS5 -> Online)
│   ├── uesp_indexer.py          # Streaming XML parser for UESP dump
│   ├── sarvam_client.py         # Sarvam AI Indic cloud client
│   └── pipeline.py              # End-to-end streaming voice pipeline CLI
├── eval/
│   ├── test_prompts.json        # Curated prompts across 13 events, lore & OOC traps
│   ├── benchmark.py             # Quantitative evaluation harness
│   └── results/                 # Exported latency_breakdown.csv, persona_evaluation.csv
└── tests/
    └── test_companion.py        # Comprehensive test suite (12 unit tests)
```

---

## 🚀 Quickstart & Execution

### 1. Prerequisites
- Linux (CachyOS / Arch / Ubuntu) with Python 3.10+.
- LM Studio running locally at `http://127.0.0.1:1234` with Vulkan GPU offload enabled and Gemma 4 E2B loaded.
- `ffmpeg` installed for audio handling:
  ```bash
  sudo pacman -S ffmpeg   # CachyOS / Arch Linux
  ```

### 2. Installation
```bash
git clone https://github.com/your-username/skyrim-hindi-companion.git
cd skyrim-hindi-companion
pip install -r requirements.txt
```

### 3. Running Unit Tests
Execute the full test suite verifying persona budgeting, output validation, sentence chunking, FIFO queues, and pipeline telemetry:
```bash
python3 -m unittest discover -s tests -v
```

### 4. Interactive Voice Pipeline (`src/pipeline.py`)

#### Interactive Text Mode (with Event Selection)
```bash
python3 src/pipeline.py --mode text --event generic_conversation
```
- Type prompts in Hindi or Hinglish.
- Dynamically switch events in the chat by typing `:event <event_name>` (e.g. `:event missed_attack`, `:event idle`).

#### Push-to-Talk Microphone Mode with Audio Playback
```bash
python3 src/pipeline.py --mode mic --play
```
Press `[ENTER]` to start speaking in Hindi, speak, and press `[ENTER]` again. The pipeline transcribes, generates Sarkar Bhaiya's response via streaming LLM, and plays audio through your speakers without chunk overlap.

---

## 🎙️ Optional: Sarvam AI Indic Cloud Mode

You can optionally route speech recognition and speech synthesis through [Sarvam AI](https://www.sarvam.ai)'s production Indic models (**Bulbul v4-flash** TTS and **Saaras v4** ASR).

> [!IMPORTANT]
> Never hardcode or commit API keys. Export your key securely as an environment variable:
```bash
export SARVAM_API_KEY="your_actual_sarvam_api_key_here"
```

Run the pipeline using Sarvam AI:
```bash
python3 src/pipeline.py --mode text --sarvam
# Or with push-to-talk mic and audio playback:
python3 src/pipeline.py --mode mic --sarvam --play --speaker arvind
```

---

## 📊 Evaluation & Latency Telemetry (`eval/benchmark.py`)

Run the automated evaluation benchmark across the 13 gameplay events, lore recall, and OOC injections:
```bash
python3 eval/benchmark.py
```

### Verified Benchmark Output
```text
======================================================================
     👑 SARKAR BHAIYA (सरकार भैया) EVALUATION BENCHMARK SUMMARY     
======================================================================
• Persona Consistency: 96.0% across Skyrim event & lore prompts
• OOC Trap Defense Rate: 100.0% (prompt injections deflected in-character)
• Average End-to-End Latency: 641.1 ms (P90: 762.0 ms, P95: 776.9 ms)
• Streaming TTFT: 0.4 ms | First Audible Audio: 281.0 ms
• Mode: Headless Offline Fallback / Live LM Studio GPU Inference
======================================================================
Results exported to: eval/results/
  - latency_breakdown.csv
  - persona_evaluation.csv
  - benchmark_summary.json
======================================================================
```

---

## 🛡️ Security & Git Hygiene
- API keys are read strictly from environment variables (`SARVAM_API_KEY`).
- `.env`, `*.key`, `config.local.yaml`, and database/audio binaries are covered by `.gitignore`.
- LLM outputs are validated before playback to guarantee no leaked code, English fragments, or stage direction artifacts.
