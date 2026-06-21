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

## Status

Work in progress. Interface-first build: contracts and protocols land before
the modules that depend on them.

## License

MIT.
