package llm

import (
	"fmt"

	"github.com/go-wasmcontainers/llm/gguf"
)

// Config is the llama-architecture hyperparameters read from GGUF metadata
// (llama.* keys; gguf.md "LLM" section).
type Config struct {
	Layers, Embd, FF, Heads, KVHeads, HeadDim, Vocab, Ctx int
	RopeBase, Eps                                         float32
}

// ConfigFrom reads and checks the metadata. Only architecture "llama" with
// rope dimension equal to the head size and no rope scaling is supported.
func ConfigFrom(f *gguf.File) (Config, error) {
	if a, _ := f.Str("general.architecture"); a != "llama" {
		return Config{}, fmt.Errorf("llm: architecture %q is not supported (llama only)", a)
	}
	u := func(k string) int {
		v, _ := f.Uint("llama." + k)
		return int(v)
	}
	c := Config{Layers: u("block_count"), Embd: u("embedding_length"), FF: u("feed_forward_length"),
		Heads: u("attention.head_count"), KVHeads: u("attention.head_count_kv"), Ctx: u("context_length")}
	if c.KVHeads == 0 {
		c.KVHeads = c.Heads
	}
	if c.Layers <= 0 || c.Embd <= 0 || c.FF <= 0 || c.Heads <= 0 || c.Embd%c.Heads != 0 || c.Heads%c.KVHeads != 0 {
		return Config{}, fmt.Errorf("llm: implausible llama hyperparameters %+v", c)
	}
	c.HeadDim = c.Embd / c.Heads
	if rd := u("rope.dimension_count"); rd != 0 && rd != c.HeadDim {
		return Config{}, fmt.Errorf("llm: rope.dimension_count %d differs from the head size %d", rd, c.HeadDim)
	}
	for _, k := range []string{"llama.rope.scaling.type", "llama.rope.scaling.factor"} {
		if _, ok := f.Meta[k]; ok {
			return Config{}, fmt.Errorf("llm: rope scaling (%s) is not supported", k)
		}
	}
	rb, ok := f.Float("llama.rope.freq_base")
	if !ok {
		rb = 10000
	}
	eps, _ := f.Float("llama.attention.layer_norm_rms_epsilon")
	c.RopeBase, c.Eps = float32(rb), float32(eps)
	t, ok := f.Tensor("token_embd.weight")
	if !ok {
		return Config{}, fmt.Errorf("llm: no token_embd.weight")
	}
	c.Vocab = int(t.Dims[1])
	return c, nil
}
