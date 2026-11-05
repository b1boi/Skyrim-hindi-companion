"""
Skyrim Hindi AI Companion — Core Systems Architecture.

Pipeline modules:
- ASR: Faster-Whisper automatic speech recognition
- Transliterate: Skyrim lore and Hinglish to Devanagari text normalization
- LLM: Local companion persona inference client
- TTS: XTTS-v2 zero-shot Hindi speech synthesis & streaming chunker
- DataPrep: Audio preprocessing and LJSpeech dataset curation
- Pipeline: End-to-end voice loop integration
"""

__version__ = "1.0.0"
__author__ = "Senior ML & Audio Systems Engineer"

