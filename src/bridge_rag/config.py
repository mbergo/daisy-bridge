"""Configuration — the dev/prod keystone.

The whole "runs on a laptop AND wires the real production path" promise rests
here. There is exactly ONE code path; `BRIDGE_RAG_PROFILE` selects a
`ModelProfile` whose fields are *data* (model names, dims, thresholds). No
module is allowed to branch on the profile name or hardcode a dimension — it
reads `settings.profile.embed_dim` instead. That discipline is what keeps the
dev E2E test an honest proof of the prod architecture.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProfileName(str, Enum):
    DEV = "dev"
    PROD = "prod"


class GeneratorBackend(str, Enum):
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    STUB = "stub"  # no model deps; deterministic span-echo generator for tests


class BridgeKind(str, Enum):
    BOTTLENECK = "bottleneck"
    GATED = "gated"
    ATTENTION = "attention"
    RESIDUAL = "residual"


class Regularizer(str, Enum):
    L1 = "l1"  # Eq.10 sparsity / selectivity
    L2 = "l2"  # Eq.9 energy control
    VIB = "vib"  # variational IB KL-to-prior, closed-form I(U;X) upper bound


class ModelProfile(BaseSettings):
    """Everything that differs between laptop and datacenter, as data.

    `embed_dim`, `bottleneck_dim`, `gen_hidden_dim` are read by the bridges so
    a 384-d dev embedder and a 768-d prod embedder traverse identical code.
    """

    name: ProfileName

    # Stage A — perception (frozen embedder)
    embedder_model: str
    embed_dim: int

    # Bridge AB — sidecar span extractor
    sidecar_model: str
    bottleneck_dim: int
    max_candidates: int  # ANN top-k cap (bounds the sidecar input -> budget)
    max_spans: int  # output cap, ~100-300 tokens worth (paper)
    span_token_budget: int

    # Bridge BC + Stage C — QLoRA generator
    generator_model: str
    generator_backend: GeneratorBackend
    gen_hidden_dim: int
    qlora_rank: int

    # Distillation teacher (Step 2). In dev, teacher == student (no-op distill).
    teacher_model: str

    # Latency budget (env-relative). CI never asserts the prod number.
    sidecar_budget_ms: float
    ttft_budget_ms: float

    model_config = SettingsConfigDict(extra="forbid")


DEV_PROFILE = ModelProfile(
    name=ProfileName.DEV,
    embedder_model="sentence-transformers/all-MiniLM-L6-v2",
    embed_dim=384,
    sidecar_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    bottleneck_dim=64,
    max_candidates=32,
    max_spans=8,
    span_token_budget=300,
    generator_model="sshleifer/tiny-gpt2",
    generator_backend=GeneratorBackend.STUB,
    gen_hidden_dim=256,
    qlora_rank=8,
    teacher_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    sidecar_budget_ms=50.0,
    ttft_budget_ms=2000.0,
)

PROD_PROFILE = ModelProfile(
    name=ProfileName.PROD,
    embedder_model="intfloat/e5-large-v2",
    embed_dim=768,
    sidecar_model="BAAI/bge-reranker-v2-m3",
    bottleneck_dim=256,
    max_candidates=128,
    max_spans=64,
    span_token_budget=300,
    generator_model="Qwen/Qwen2.5-7B-Instruct",
    generator_backend=GeneratorBackend.VLLM,
    gen_hidden_dim=3584,
    qlora_rank=64,
    teacher_model="Qwen/Qwen3-8B",
    sidecar_budget_ms=0.3,
    ttft_budget_ms=400.0,
)

_PROFILES: dict[ProfileName, ModelProfile] = {
    ProfileName.DEV: DEV_PROFILE,
    ProfileName.PROD: PROD_PROFILE,
}


class Settings(BaseSettings):
    """Top-level runtime settings. Read once, cached."""

    profile_name: ProfileName = Field(default=ProfileName.DEV)

    # Bridge selection (which TensorBridge family each seam uses).
    bridge_ab: BridgeKind = BridgeKind.ATTENTION  # sidecar interface (Eq.18-20)
    bridge_bc: BridgeKind = BridgeKind.GATED  # generator conditioning (Eq.16-17)

    # Loss knobs (Eq.8). MINE is diagnostic-only — never on the backward path.
    regularizer: Regularizer = Regularizer.L1
    lambda_ab: float = 1e-3
    lambda_bc: float = 1e-3
    beta: float = 1.0  # I(U;X) - beta*I(U;Y)
    enable_mine_diagnostic: bool = False

    # LR partitioning (Stabilizer 4 — the primary lever against Eq.13 collapse).
    lr_fa: float = 1e-5
    lr_gab: float = 1e-4
    lr_fc: float = 1e-4
    lr_gbc: float = 1e-3
    grad_clip_norm: float = 1.0
    dropout_gab: float = 0.1
    dropout_gbc: float = 0.05

    corpus_path: str = "data/corpus.jsonl"
    index_path: str = "data/corpus.faiss"

    model_config = SettingsConfigDict(
        env_prefix="BRIDGE_RAG_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def profile(self) -> ModelProfile:
        return _PROFILES[self.profile_name]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Test helper: drop the cached Settings so env overrides re-read."""
    get_settings.cache_clear()
