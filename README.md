# llm

A small-LLM inference library in Go for WASI containers: a `wasip1` guest
reads a GGUF model from a read-only mount (`/models/NAME.gguf`) and generates
text on the GPU through the batched command stream of
[go-wasmcontainers/gpu](https://github.com/go-wasmcontainers/gpu)
(`wasi_webgpu_batch`), in the browser first. Design and plan:
`docs/llm-design.md` in go-wasmcontainers/gpu.

| Package | What |
|---|---|
| `gguf` | bounded GGUF v3 header parser over an `io.ReaderAt`; fuzzed, corpus committed |
| `llm` (root) | byte-level BPE tokenizer (`gpt2` model, `smollm` pre-tokenizer), llama config, CPU forward pass (the oracle and native fallback), GPU forward pass and its WGSL f32 kernels |
| `examples/generate` | the container: `run`, `check` (GPU against the CPU path), `bench` |

First model: SmolLM2-135M-Instruct Q4_0 (Q4_0, Q4_1 and Q8_0 tensors).

## Design points

- **Streaming load.** Weights go from the file to the GPU in 16 MiB pieces,
  repacked per block (18-byte Q4_0 blocks become aligned `u32` quants and
  `f32` scales); the model never sits in guest memory.
- **One submit per token.** Every dispatch of the forward pass is recorded
  into one compute pass; sampling (`argmax`) runs on the GPU and only the
  4-byte token is read back. Prompt tokens go in submits of 8.
- **Vendor-neutral f32.** No subgroups, no `shader-f16`. The workgroup size
  (64, lower if the adapter's invocation or workgroup-memory limit is lower),
  the 2-D grids and the buffer sizes come from the adapter limits the host
  reports. Reductions are tree reductions in workgroup memory: the order of
  summation depends on the workgroup size, not on the hardware. RoPE angles
  are computed on the CPU in double precision, so no adapter's `sin`/`cos`
  accuracy enters. Only Apple Metal has been run; no NVIDIA or AMD result is
  claimed.

## Tests

```
go test ./...                                  # parser, refusals, fuzz corpus, tokenizer-free units
LLM_MODELS=DIR go test -run Oracle -v .        # needs llama-server (the go-pkgx llama.cpp bottle) and
                                               # DIR/SmolLM2-135M-Instruct-Q4_0.gguf (pinned in go-wasmcontainers-llm-corpus)
```

Model files are never committed; tests that need them skip without
`LLM_MODELS`. The GPU oracle runs in the browser from go-wasmcontainers/gpu
(`LLM_REPO=... go test -tags e2e ./e2e -run TestLLMOracle`).

### Oracle tolerances (vendor-sensitive tests are marked)

- Tokenizer: ids identical to llama.cpp's `/tokenize` on the fixed prompts.
- CPU path against llama.cpp (`TestOracleLlamaCpp`, one thread, greedy):
  a token differing from llama.cpp's is accepted only at a near-tie of
  llama.cpp's own distribution (gap at most 0.25 nats) after at least 6
  equal tokens; step-1 log-probability differences against its top-10 stay
  within 0.5 nats. Measured: the ggml-exact mode (Q8_0 activations, f16 KV
  cache) does not reproduce llama.cpp to 1e-3, so no tighter bound is
  claimed.
- GPU against the CPU f32 path (**vendor-sensitive**): logits within 1e-3
  relative to the largest logit, and equal greedy tokens. Fused
  multiply-adds and a different `exp` on another vendor move the last bits.
- GPU against llama.cpp (**vendor-sensitive**): as the CPU path above.
