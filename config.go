package llm

import (
	"fmt"

	"github.com/go-wasmcontainers/llm/gguf"
)

// Config is the llama-family hyperparameters read from GGUF metadata
// (<arch>.* keys; gguf.md "LLM" section).
type Config struct {
	Layers, Embd, FF, Heads, KVHeads, HeadDim, Vocab, Ctx int
	RopeBase, Eps                                         float32
	// Arch is general.architecture: "llama" or "qwen2".
	Arch string
	// RopeNeox: the rotary pairs are (i, i+HeadDim/2) (llama.cpp rope type
	// NEOX, which qwen2 uses) instead of adjacent elements (NORM, llama).
	RopeNeox bool
	// Bias: the Q, K and V projections have biases (qwen2).
	Bias bool
	// TiedOutput: no output.weight, the token embedding is the output matrix.
	TiedOutput bool
}

// ConfigFrom reads and checks the metadata. Architectures "llama" (rope type
// NORM) and "qwen2" (rope type NEOX, QKV biases: llama.cpp src/models/qwen2.cpp,
// src/llama-model.cpp), with rope dimension equal to the head size and no rope
// scaling.
func ConfigFrom(f *gguf.File) (Config, error) {
	arch, _ := f.Str("general.architecture")
	if arch != "llama" && arch != "qwen2" {
		return Config{}, fmt.Errorf("llm: architecture %q is not supported (llama and qwen2 only)", arch)
	}
	u := func(k string) int {
		v, _ := f.Uint(arch + "." + k)
		return int(v)
	}
	c := Config{Arch: arch, RopeNeox: arch == "qwen2", Layers: u("block_count"), Embd: u("embedding_length"), FF: u("feed_forward_length"),
		Heads: u("attention.head_count"), KVHeads: u("attention.head_count_kv"), Ctx: u("context_length")}
	if c.KVHeads == 0 {
		c.KVHeads = c.Heads
	}
	if c.Layers <= 0 || c.Embd <= 0 || c.FF <= 0 || c.Heads <= 0 || c.Embd%c.Heads != 0 || c.Heads%c.KVHeads != 0 {
		return Config{}, fmt.Errorf("llm: implausible %s hyperparameters %+v", arch, c)
	}
	c.HeadDim = c.Embd / c.Heads
	if rd := u("rope.dimension_count"); rd != 0 && rd != c.HeadDim {
		return Config{}, fmt.Errorf("llm: rope.dimension_count %d differs from the head size %d", rd, c.HeadDim)
	}
	for _, k := range []string{arch + ".rope.scaling.type", arch + ".rope.scaling.factor"} {
		if _, ok := f.Meta[k]; ok {
			return Config{}, fmt.Errorf("llm: rope scaling (%s) is not supported", k)
		}
	}
	rb, ok := f.Float(arch + ".rope.freq_base")
	if !ok {
		rb = 10000
	}
	eps, _ := f.Float(arch + ".attention.layer_norm_rms_epsilon")
	c.RopeBase, c.Eps = float32(rb), float32(eps)
	t, ok := f.Tensor("token_embd.weight")
	if !ok {
		return Config{}, fmt.Errorf("llm: no token_embd.weight")
	}
	c.Vocab = int(t.Dims[1])
	_, hasOut := f.Tensor("output.weight")
	c.TiedOutput = !hasOut
	_, c.Bias = f.Tensor("blk.0.attn_q.bias")
	if c.Bias != (arch == "qwen2") {
		return Config{}, fmt.Errorf("llm: %s with biases %v is not supported", arch, c.Bias)
	}
	return c, nil
}
