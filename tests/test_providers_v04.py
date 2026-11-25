"""Tests for Batch 5 providers: Mistral FIM, xAI Grok, Perplexity, OpenAI Audio/Image, HF."""

from __future__ import annotations

from llm_tape.parsers import parse_llm_interaction
from llm_tape.pricing import estimate_cost_usd


def test_mistral_fim_and_pricing() -> None:
    summary = parse_llm_interaction(
        url="https://api.mistral.ai/v1/fim/completions",
        request_body={"model": "codestral-latest", "prompt": "def add(a, b):"},
        response_body={
            "id": "cmpl-1",
            "model": "codestral-latest",
            "choices": [{"message": {"role": "assistant", "content": " return a + b"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )
    assert summary.provider == "openai"
    assert summary.model == "codestral-latest"
    assert summary.text == " return a + b"
    assert summary.estimated_cost_usd > 0.0
    assert estimate_cost_usd("ministral-8b-latest", 1_000_000, 1_000_000) == 0.20
    assert estimate_cost_usd("ministral-3b-latest", 1_000_000, 1_000_000) == 0.08


def test_xai_grok_detection_and_pricing() -> None:
    summary = parse_llm_interaction(
        url="https://api.x.ai/v1/chat/completions",
        request_body={"model": "grok-3-mini", "messages": [{"role": "user", "content": "hi"}]},
        response_body={
            "model": "grok-3-mini",
            "choices": [{"message": {"role": "assistant", "content": "hello from grok"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 500},
            },
        },
    )
    assert summary.provider == "openai"
    assert summary.model == "grok-3-mini"
    assert summary.text == "hello from grok"
    assert summary.usage.cached_input_tokens == 500
    assert summary.estimated_cost_usd > 0.0
    assert estimate_cost_usd("grok-3", 1_000_000, 1_000_000) == 18.0
    assert estimate_cost_usd("grok-2-1212", 1_000_000, 1_000_000) == 12.0


def test_perplexity_detection_and_pricing() -> None:
    summary = parse_llm_interaction(
        url="https://api.perplexity.ai/chat/completions",
        request_body={"model": "sonar-pro", "messages": [{"role": "user", "content": "news"}]},
        response_body={
            "model": "sonar-pro",
            "choices": [{"message": {"role": "assistant", "content": "today's news"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
        },
    )
    assert summary.provider == "openai"
    assert summary.model == "sonar-pro"
    assert summary.estimated_cost_usd == 0.018
    assert estimate_cost_usd("sonar", 1_000_000, 1_000_000) == 2.0
    assert estimate_cost_usd("sonar-reasoning", 1_000_000, 1_000_000) == 6.0


def test_openai_audio_and_image_detection() -> None:
    audio = parse_llm_interaction(
        url="https://proxy.internal/v1/audio/transcriptions",
        request_body={"model": "whisper-1"},
        response_body={"text": "hello world transcription"},
    )
    assert audio.provider == "openai"
    assert audio.text == "hello world transcription"

    image = parse_llm_interaction(
        url="https://proxy.internal/v1/images/generations",
        request_body={"model": "dall-e-3", "prompt": "a cat"},
        response_body={
            "data": [{"revised_prompt": "A detailed orange tabby cat", "url": "https://x"}]
        },
    )
    assert image.provider == "openai"
    assert image.text == "A detailed orange tabby cat"


def test_huggingface_router_detection() -> None:
    summary = parse_llm_interaction(
        url="https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions",
        request_body={"model": "meta-llama/Llama-3.3-70B-Instruct"},
        response_body={
            "choices": [{"message": {"role": "assistant", "content": "hi from hf"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    assert summary.provider == "openai"
    assert summary.text == "hi from hf"

    legacy = parse_llm_interaction(
        url="https://api-inference.huggingface.co/models/gpt2",
        request_body={"inputs": "hello"},
        response_body={"text": "hello world"},
    )
    assert legacy.provider == "openai"
    assert legacy.text == "hello world"
