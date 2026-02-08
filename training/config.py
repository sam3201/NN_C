from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: int = 60


@dataclass
class ConsensusConfig:
    min_similarity: float = 0.72
    min_votes: int = 2
    max_candidates: int = 4


@dataclass
class DistillationConfig:
    tasks_path: str
    output_path: str
    n_per_teacher: int = 1
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)


@dataclass
class TrainingConfig:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    max_seq_len: int = 1024
    batch_size: int = 2
    grad_accum: int = 8
    num_epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None


@dataclass
class RegressionConfig:
    tasks_path: str
    provider_spec: str
    min_pass_rate: float = 0.7
    max_examples: Optional[int] = None
    timeout_s: int = 60
    strict: bool = True
    output_json: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
