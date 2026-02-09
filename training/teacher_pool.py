from __future__ import annotations

import json
import logging
from pathlib import Path
import os
import time
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("teacher_pool")


def _ensure_logger() -> None:
    if logger.handlers:
        return
    level = os.getenv("SAM_TEACHER_LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("SAM_TEACHER_LOG_FILE", "logs/teacher_pool.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))


_ensure_logger()


@dataclass
class Candidate:
    provider: str
    model: str
    prompt: str
    response: str
    latency_s: float


class Provider:
    def __init__(self, name: str, model: str, temperature: float = 0.2, max_tokens: int = 512, timeout_s: int = 60):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def generate(self, prompt: str):
        raise NotImplementedError

    def _log_start(self, prompt_len: int) -> None:
        logger.info(
            "Teacher request start provider=%s model=%s prompt_len=%d max_tokens=%d temp=%.2f timeout_s=%d",
            self.name,
            self.model,
            prompt_len,
            self.max_tokens,
            self.temperature,
            self.timeout_s,
        )

    def _log_done(self, latency_s: float, response_len: int, status_code: int | None = None) -> None:
        logger.info(
            "Teacher request done provider=%s model=%s status=%s latency_s=%.2f response_len=%d",
            self.name,
            self.model,
            status_code if status_code is not None else "n/a",
            latency_s,
            response_len,
        )

    def _log_fail(self, latency_s: float, error: Exception) -> None:
        logger.warning(
            "Teacher request failed provider=%s model=%s latency_s=%.2f error=%s",
            self.name,
            self.model,
            latency_s,
            error,
        )


class OllamaProvider(Provider):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs: Any):
        super().__init__("ollama", model, **kwargs)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(self, prompt: str):
        self._log_start(len(prompt))
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        start = time.time()
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            latency = time.time() - start
            self._log_done(latency, len(response_text), resp.status_code)
            return response_text, latency
        except Exception as exc:
            self._log_fail(time.time() - start, exc)
            raise


class OpenAIProvider(Provider):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
        super().__init__("openai", model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def generate(self, prompt: str):
        self._log_start(len(prompt))
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        start = time.time()
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            latency = time.time() - start
            self._log_done(latency, len(content), resp.status_code)
            return content, latency
        except Exception as exc:
            self._log_fail(time.time() - start, exc)
            raise


class OpenRouterProvider(Provider):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
        super().__init__("openrouter", model, **kwargs)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

    def generate(self, prompt: str):
        self._log_start(len(prompt))
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        start = time.time()
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            latency = time.time() - start
            self._log_done(latency, len(content), resp.status_code)
            return content, latency
        except Exception as exc:
            self._log_fail(time.time() - start, exc)
            raise


class HFLocalProvider(Provider):
    def __init__(self, model: str, adapter_path: Optional[str] = None,
                 device_map: Optional[str] = None, torch_dtype: Optional[str] = None, **kwargs: Any):
        super().__init__("hf", model, **kwargs)
        self.adapter_path = adapter_path
        self.device_map = device_map or os.getenv("SAM_HF_DEVICE_MAP", "auto")
        self.torch_dtype_name = torch_dtype or os.getenv("SAM_HF_DTYPE", "float16")
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:
                raise RuntimeError(f"HFLocalProvider requires torch+transformers: {exc}") from exc
            dtype = getattr(torch, self.torch_dtype_name, torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=dtype,
                device_map=self.device_map,
            )
            if self.adapter_path:
                try:
                    from peft import PeftModel
                except Exception as exc:
                    raise RuntimeError(f"HFLocalProvider requires peft for adapters: {exc}") from exc
                adapter_path = str(Path(self.adapter_path).expanduser())
                model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def generate(self, prompt: str):
        self._log_start(len(prompt))
        start = time.time()
        try:
            self._ensure_loaded()
            assert self._model is not None and self._tokenizer is not None
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            gen_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            text = self._tokenizer.decode(output[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):].lstrip()
            latency = time.time() - start
            self._log_done(latency, len(text))
            return text, latency
        except Exception as exc:
            self._log_fail(time.time() - start, exc)
            raise


def build_provider(spec: str, temperature: float = 0.2, max_tokens: int = 512, timeout_s: int = 60) -> Provider:
    if ":" not in spec:
        raise ValueError("Provider spec must be like 'ollama:mistral:latest' or 'openrouter:model'")
    provider, model = spec.split(":", 1)
    if provider == "ollama":
        return OllamaProvider(model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
    if provider == "openai":
        return OpenAIProvider(model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
    if provider == "openrouter":
        return OpenRouterProvider(model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
    if provider == "hf":
        adapter_path = None
        base_model = model
        if "@" in model:
            base_model, adapter_path = model.split("@", 1)
        return HFLocalProvider(base_model, adapter_path=adapter_path,
                               temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
    raise ValueError(f"Unknown provider: {provider}")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


class TeacherPool:
    def __init__(self, providers: List[Provider], min_similarity: float = 0.72, min_votes: int = 2):
        if not providers:
            raise ValueError("TeacherPool requires at least one provider")
        self.providers = providers
        self.min_similarity = min_similarity
        self.min_votes = min_votes

    def generate(self, prompt: str, n_per_teacher: int = 1) -> List[Candidate]:
        candidates: List[Candidate] = []
        for provider in self.providers:
            for _ in range(n_per_teacher):
                try:
                    response, latency = provider.generate(prompt)
                except Exception as exc:
                    logger.warning(
                        "Teacher provider failed provider=%s model=%s error=%s",
                        provider.name,
                        provider.model,
                        exc,
                    )
                    continue
                candidates.append(Candidate(provider=provider.name, model=provider.model,
                                            prompt=prompt, response=response, latency_s=latency))
        return candidates

    def consensus_filter(self, candidates: List[Candidate]) -> List[Candidate]:
        if len(candidates) <= 1:
            return candidates
        filtered: List[Candidate] = []
        for i, cand in enumerate(candidates):
            votes = 0
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                if similarity(cand.response, other.response) >= self.min_similarity:
                    votes += 1
            if votes >= self.min_votes:
                filtered.append(cand)
        return filtered if filtered else candidates
