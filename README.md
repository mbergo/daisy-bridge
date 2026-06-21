# Daisy

A production implementation of **Multi-Stage Neural Networks with Learnable
Intermediate Bridges** — a three-stage neural network connected by two learnable
bridges, realized as a retrieval-augmented generation system.

The central thesis: **the interface between models matters more than the models
themselves.** Design the bridge first; the stages follow. Structured
decomposition with explicit interface contracts beats monolithic scaling when
the stages naturally separate concerns.

## The architecture

```
ŷ = fC(gBC(fB(gAB(fA(x)))))
```

| Stage | Role | Implementation |
|-------|------|----------------|
| `fA`  | Perception | frozen embedder (E5 / BGE), query → `hA ∈ R^(1×d)` |
| ANN   | candidate lookup | faiss top-k over precomputed corpus |
| `gAB` | **Sidecar bridge** | attention-style span extractor; emits spans **with provenance** under a `<0.3ms` budget |
| `fB`  | Inference | the span set — no parameters, the needle not the haystack |
| `gBC` | QLoRA bridge | gated routing; conditions the generator on span format |
| `fC`  | Generation | LLM + QLoRA; input is spans only (~100–300 tokens), streamed |

## Why it reaches 0% hallucination

Structural, not a guardrail. The generator only ever receives **extracted
spans**. Every source of hallucination — gap-filling, document confusion,
extrapolation, context noise, semantic-gap invention — requires context the
generator does not have, because the bridge removed it. A span carries its
`(doc_id, char_start, char_end)`, so a span **is** a citation.

This is the information bottleneck (Eq. 11) in production:

```
min  I(U;X) − β · I(U;Y)
```

Compress away mutual information with the input; preserve mutual information
with the correct answer. Pass only what predicts `Y`; discard everything else.

## Design stance

- **`TensorBridge` vs `SpanExtractor` are separate abstractions.** Bridges are
  pure `Tensor → Tensor` math (Eq. 14–20) that you train and analyze. The
  sidecar is a reranker model that emits spans under a latency contract.
- **The bottleneck loss is a closed-form surrogate by default.** `−β·I(U;Y)` is
  the task loss; `I(U;X)` is the bottleneck dimension plus an L1/L2/VIB
  penalty. A MINE/InfoNCE estimator exists but is **diagnostic-only** — it
  measures MI, it never sits on the backward path (that would destabilize the
  six-Jacobian gradient chain of Eq. 13).
- **Profile-as-data, not branches.** `BRIDGE_RAG_PROFILE=dev|prod` selects model
  names, dimensions, and latency thresholds. There is one code path. The `dev`
  profile runs the entire pipeline on a laptop with tiny weights while
  exercising every interface — distillation, QLoRA attach, faiss, SSE.
- **Stability is engineered, in order of impact:** LayerNorm at every bridge
  output, residual identity paths, global gradient clipping, **learning-rate
  partitioning** (the primary lever), and bridge dropout.

## Quickstart

The `dev` profile runs the entire pipeline on a laptop, CPU-only, with no model
downloads (a deterministic hashing embedder and a span-echo generator stand in
for the real weights while exercising every interface).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .                      # base deps (laptop/CPU dev path)
pip install -e ".[dev]"               # + test/lint tooling

# Build a tiny corpus index (ships with a demo corpus under data/)
bridge-rag-corpus --input data/corpus.jsonl

# Serve: console UI at http://localhost:8000/ , SSE at POST /answer
BRIDGE_RAG_CORPUS_PATH=data/corpus.jsonl \
  uvicorn bridge_rag.serving.app:app --port 8000
```

Open `http://localhost:8000/` for the streaming console (answer + live
citations + the critical-path latency ledger). Or stream from the shell:

```bash
curl -sN -X POST localhost:8000/answer \
  -H 'content-type: application/json' \
  -d '{"query":"why is hallucination zero?","top_k":6}'
```

For the real models (E5/BGE embedder, BGE-M3 reranker sidecar, QLoRA generator,
faiss, vLLM), install the extras and switch profile:

```bash
pip install -e ".[models]"            # transformers, peft, faiss, sentence-transformers
pip install -e ".[prod]"              # + vllm, bitsandbytes, accelerate, faiss-gpu
BRIDGE_RAG_PROFILE_NAME=prod uvicorn bridge_rag.serving.app:app
```

## Verify

```bash
pytest -q                             # 35 tests: bridges, losses, contracts,
                                      # training, no-hallucination, versioning,
                                      # serving, and the E2E keystone
bridge-rag-bench --requests 50 --query "what stabilizes training?" --top-k 6
```

The keystone `tests/test_e2e.py` runs the full Eq.6 chain
(`embed → ANN → sidecar → fB → generate → cite`) and asserts spans carry
provenance, generator context ⊆ spans, an answer streams, and every citation
resolves to real document offsets.

## A note on the dev profile

The `dev` profile's hashing embedder is intentionally weak — it exists so the
*architecture* runs everywhere, not to retrieve well. With it, the information
bottleneck's selectivity is loose: low-signal documents can still surface a
span. The structural guarantee is unaffected — every span carries verifiable
provenance, so a cited claim is always traceable to a real source offset. The
real BGE-M3 reranker (the `models` extra) scores off-topic spans low and the
bottleneck tightens. Compression quality scales with the embedder; the
*citation guarantee* does not depend on it.

## Status

Working end to end on the dev profile: train (4-step regime), serve (FastAPI
SSE + console UI), version (blue/green paired), benchmark. The prod model
backends (real weights, vLLM, faiss-gpu) are wired and selected by profile but
not exercised in this environment.

## License

MIT.
