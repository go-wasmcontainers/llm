//go:build !js

// Command generate is a WASI container that generates text with a GGUF llama
// model on the GPU, through the batched command stream (wasi_webgpu_batch).
//
//	GOOS=wasip1 GOARCH=wasm go build -o generate.wasm ./examples/generate
//	wctr run --gpu --gpu-memory=512m --model smollm2-135m-instruct-q4_0 IMAGE [MODE] ...
//
// MODE:
//
//	run   [-n N] [-ctx C] [-model FILE] PROMPT   greedy generation on the GPU
//	check [-n N] [-model FILE] PROMPT            the same prompt on the GPU and on the
//	                                             CPU path (f32): logits at step 1 and the
//	                                             first N token ids
//	bench [-rounds R] [-p P] [-n N] [-model FILE] R rounds of P-token prefill and N-token decode
//
// Exit 2 (printing "no GPU") when the container has no GPU. Natively (not
// wasip1) only -cpu works: there is no GPU adapter.
package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/go-wasmcontainers/gpu/wasigpu/batchsink"

	llm "github.com/go-wasmcontainers/llm"
	"github.com/go-wasmcontainers/llm/gguf"
)

const defaultModel = "/models/smollm2-135m-instruct-q4_0.gguf"

func die(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		die(fmt.Errorf("usage: generate run|check|bench [flags] [PROMPT]"))
	}
	mode, args := os.Args[1], os.Args[2:]
	fs := flag.NewFlagSet(mode, flag.ExitOnError)
	n := fs.Int("n", 32, "tokens to generate")
	ctx := fs.Int("ctx", 512, "KV cache length")
	model := fs.String("model", defaultModel, "GGUF file")
	rounds := fs.Int("rounds", 5, "bench rounds")
	p := fs.Int("p", 128, "bench prompt length")
	fs.Parse(args)
	prompt := strings.Join(fs.Args(), " ")

	fh, err := os.Open(*model)
	if err != nil {
		die(err)
	}
	st, _ := fh.Stat()
	t0 := time.Now()
	f, err := gguf.Parse(fh, uint64(st.Size()))
	if err != nil {
		die(err)
	}
	tok, err := llm.NewTokenizer(f)
	if err != nil {
		die(err)
	}
	fmt.Printf("header: %d tensors, %d bytes, parsed in %.0f ms\n", len(f.Tensors), st.Size(), ms(time.Since(t0)))

	s, err := batchsink.Open()
	if err != nil {
		fmt.Println("no GPU:", err)
		os.Exit(2)
	}
	defer s.Close()
	c := s.Caps()
	lim := llm.Limits{
		MaxInvocationsPerWorkgroup: c.Device.MaxComputeInvocationsPerWorkgroup, MaxWorkgroupStorage: c.Device.MaxComputeWorkgroupStorageSize,
		MaxWorkgroupsPerDimension: c.Device.MaxComputeWorkgroupsPerDimension, MaxStorageBuffers: c.Device.MaxStorageBuffersPerShaderStage,
		MaxBindingSize: c.Device.MaxStorageBufferBindingSize, MaxBufferSize: c.Device.MaxStorageBufferBindingSize, MemoryBudget: c.MaxMemory,
	}
	fmt.Printf("limits: workgroup size %d, %d invocations, %d B shared, %d groups/dim, binding %d, memory %d\n",
		lim.WorkgroupSize(), lim.MaxInvocationsPerWorkgroup, lim.MaxWorkgroupStorage, lim.MaxWorkgroupsPerDimension, lim.MaxBindingSize, lim.MemoryBudget)
	t0 = time.Now()
	g, err := llm.NewGPU(s.Device("llm"), lim, f, fh, *ctx)
	if err != nil {
		die(err)
	}
	fmt.Printf("load: %.0f ms, GPU memory %d bytes, %d submits\n", ms(time.Since(t0)), g.MemoryUsed(), g.Stats.Submits)

	switch mode {
	case "run":
		ids := tok.Encode(prompt)
		var out []int32
		tm, err := g.Generate(ids, *n, tok.EOS, func(t int32) bool { out = append(out, t); return true })
		if err != nil {
			die(err)
		}
		fmt.Printf("prompt: %v\ntokens: %v\ntext: %q\n", ids, out, tok.Decode(out))
		report(tm)
	case "check":
		check(g, f, fh, tok, prompt, *n, *ctx)
	case "trace":
		trace(g, f, fh, tok, prompt, *ctx)
	case "bench":
		bench(g, tok, *rounds, *p, *n)
	default:
		die(fmt.Errorf("unknown mode %q", mode))
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func report(tm llm.Timing) {
	fmt.Printf("prefill: %d tokens in %.1f ms = %.1f tok/s\n", tm.PromptTokens, ms(tm.Prefill), float64(tm.PromptTokens)/tm.Prefill.Seconds())
	fmt.Printf("decode: %d tokens in %.1f ms = %.1f tok/s (%d submits, %d dispatches in all)\n", tm.Generated, ms(tm.Decode),
		float64(tm.Generated)/tm.Decode.Seconds(), tm.Submits, tm.Dispatches)
	if tm.Generated > 1 {
		fmt.Printf("decode per token: %.2f ms recording in the guest, %.2f ms in submit (crossing, replay, GPU, read-back)\n",
			ms(tm.Record)/float64(tm.Generated-1), ms(tm.Submit)/float64(tm.Generated-1))
	}
}

func check(g *llm.GPU, f *gguf.File, r io.ReaderAt, tok *llm.Tokenizer, prompt string, n, ctx int) {
	ids := tok.Encode(prompt)
	gl, err := g.PrefillLogits(ids)
	if err != nil {
		die(err)
	}
	cpu, err := llm.LoadCPU(f, r, ctx)
	if err != nil {
		die(err)
	}
	t0 := time.Now()
	var cl []float32
	for _, t := range ids {
		cl = cpu.Forward(t)
	}
	var maxAbs, maxRef float64
	for i := range cl {
		maxAbs = math.Max(maxAbs, math.Abs(float64(gl[i]-cl[i])))
		maxRef = math.Max(maxRef, math.Abs(float64(cl[i])))
	}
	fmt.Printf("logits: max |gpu-cpu| = %.3e, max |cpu| = %.3f, relative %.3e, argmax gpu %d cpu %d (cpu step: %.0f ms)\n",
		maxAbs, maxRef, maxAbs/maxRef, llm.Argmax(gl), llm.Argmax(cl), ms(time.Since(t0)))
	var got []int32
	if _, err := g.Generate(ids, n, -1, func(t int32) bool { got = append(got, t); return true }); err != nil {
		die(err)
	}
	next := llm.Argmax(cl)
	var want []int32
	for len(want) < n {
		want = append(want, next)
		next = llm.Argmax(cpu.Forward(next))
	}
	eq := 0
	for eq < len(got) && eq < len(want) && got[eq] == want[eq] {
		eq++
	}
	fmt.Printf("gpu tokens: %v\ncpu tokens: %v\nequal prefix: %d of %d\n", got, want, eq, n)
}

func bench(g *llm.GPU, tok *llm.Tokenizer, rounds, p, n int) {
	base := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 40))
	if len(base) < p {
		die(fmt.Errorf("bench prompt of %d tokens needs more text", p))
	}
	prompt := base[:p]
	var pre, dec []float64
	for i := 0; i < rounds+1; i++ { // the first round warms up pipelines
		tm, err := g.Generate(prompt, n, -1, func(int32) bool { return true })
		if err != nil {
			die(err)
		}
		if i == 0 {
			continue
		}
		pre = append(pre, float64(tm.PromptTokens)/tm.Prefill.Seconds())
		dec = append(dec, float64(tm.Generated)/tm.Decode.Seconds())
		fmt.Printf("round %d: prefill %.1f tok/s, decode %.1f tok/s\n", i, pre[len(pre)-1], dec[len(dec)-1])
		report(tm)
	}
	slices.Sort(pre)
	slices.Sort(dec)
	fmt.Printf("bench p=%d n=%d rounds=%d: prefill median %.1f (min %.1f max %.1f) tok/s, decode median %.1f (min %.1f max %.1f) tok/s\n",
		p, n, rounds, pre[len(pre)/2], pre[0], pre[len(pre)-1], dec[len(dec)/2], dec[0], dec[len(dec)-1])
}

// trace compares the residual stream of the GPU and the CPU path after
// 0, 1, 2, 5, 15 and all layers, for the first token of the prompt: where
// the two first disagree is where the defect is.
func trace(g *llm.GPU, f *gguf.File, r io.ReaderAt, tok *llm.Tokenizer, prompt string, ctx int) {
	ids := tok.Encode(prompt)[:1]
	cpu, err := llm.LoadCPU(f, r, ctx)
	if err != nil {
		die(err)
	}
	for _, l := range []int{-1, 1, 2, 5, 15, 30} {
		g.Layers, cpu.MaxLayers = l, l
		cpu.Reset()
		cpu.Forward(ids[0])
		want := cpu.LastX
		if _, err := g.PrefillLogits(ids); err != nil && l == 30 {
			die(err)
		}
		got, err := g.Peek("x", len(want))
		if err != nil {
			die(err)
		}
		var maxAbs, maxRef float64
		for i := range want {
			maxAbs = math.Max(maxAbs, math.Abs(float64(got[i]-want[i])))
			maxRef = math.Max(maxRef, math.Abs(float64(want[i])))
		}
		st, _ := g.Peek("state", 2)
		tk, _ := g.Peek("tokens", 2)
		fmt.Printf("state bits %x %x tokens bits %x %x\n", math.Float32bits(st[0]), math.Float32bits(st[1]), math.Float32bits(tk[0]), math.Float32bits(tk[1]))
		fmt.Printf("layers %d: x max |gpu-cpu| = %.3e (max |x| %.3f); gpu %v cpu %v\n", l, maxAbs, maxRef, got[:4], want[:4])
	}
}
