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

### Oracle (vendor-sensitive tests are marked)

- Tokenizer: ids identical to llama.cpp's `/tokenize` on the fixed prompts.
- **Strict oracle: llama.cpp on the same weights dequantised to F32.**
  `llama-quantize --allow-requantize SmolLM2-135M-Instruct-Q4_0.gguf OUT.gguf F32`
  (llama.cpp's own tool, so the weights are identical) run with
  `--cache-type-k f32 --cache-type-v f32 -fa off`: f32 activations, f32
  cache. `TestOracleDequantizedF32`: the Go CPU f32 path gives **32 of 32
  greedy tokens on all four prompts** and step-1 log-probability differences
  of at most 1.4e-5 nats (tolerance 1e-4). The GPU container is held to the
  same tokens (**vendor-sensitive**: it differs from the CPU path by 1.6e-6
  relative in logits on Apple Metal; another vendor's fused multiply-adds and
  `exp` move the last bits, and a token flips only at a gap that small).
- **Why llama.cpp's Q4_0 CPU run differs (about 0.1 to 0.4 nats at step 1).**
  On CPU ggml quantises the *activations* to Q8_0 (Q8_1 for Q4_1 weights)
  before every quantised matrix product (`vec_dot_q4_0_q8_0`), our f32 path
  does not. The Go CPU path can emulate it (`ExactGGML`: Q8_0/Q8_1 activations,
  f16 KV cache) and that is **not** enough to match to 1e-3: such a path
  amplifies rounding by about 10^6 (`TestQuantisedActivationsAmplifyRounding`:
  summing the RMSNorm squares in float64 instead of float32, a 1e-7 change,
  moves the emulated logits by 0.12 to 0.24 nats and the f32 path by 1e-5), so
  only a bit-exact port of ggml's CPU kernels could match. The default f16 KV
  cache adds about 1e-3 to 5e-3. So the Q4_0 run is compared only
  loosely (`TestOracleLlamaCpp`): a token differing is accepted at a near-tie
  of llama.cpp's own distribution (at most 0.25 nats, after 6 equal tokens)
  and step-1 differences within 0.5 nats; on two of four prompts the f32
  path diverges from it at the 10th and 22nd token.

## Results (2026-11-03, Apple M4 Max, Chromium, Metal, f32 kernels)

SmolLM2-135M-Instruct Q4_0, one container, greedy, 128-token prompt and 128
generated tokens, five interleaved rounds against llama.cpp 0.8.0 on one CPU
core (`llama-bench -t 1 -ngl 0`, the go-pkgx bottle, openssl 3.5.8), 1-minute
load 6.4 to 7.7:

| | prefill tok/s (p50, min-max) | decode tok/s (p50, min-max) |
|---|---|---|
| container, WebGPU, f32 | 311 (302-318) | 171 (160-176) |
| llama.cpp CPU, one core | 1286 (1278-1289) | 529 (521-531) |
| WebLLM q4f32_1 (baseline, go-wasmcontainers/gpu docs/llm-baseline) | 2656 | 224 |

**Slower than the one-core CPU reference: an open defect, not yet optimised
(faithful port first).** Where the decode time goes (per token, 405 dispatches):
0.13-0.2 ms recording in the guest, 3.8 ms (short context) to 5.5-6 ms (context
of 128 to 192) in `submit`, of which about 0.4 ms is the crossing (measured
separately, docs/llm-design.md). So the GPU work itself, about 9 us per
dispatch of tiny kernels (a 576-wide row is 72 words for 64 threads), and an
attention kernel with one workgroup per head that walks the context serially,
are the cause. Prefill is a decode pass per prompt token (8 per submit):
no matrix-matrix kernel yet.

Correctness: on four fixed prompts the container's 32 greedy tokens equal
llama.cpp's on two; the other two diverge at the 10th and 22nd token at a
near-tie of llama.cpp's own distribution (0.16 and 0.12 nats), exactly as the Go
CPU path (f32) does. GPU against CPU path: logits within 1.6e-6 relative,
16 of 16 tokens equal.
