package llm

import (
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/go-wasmcontainers/llm/gguf"
)

// CPU is the forward pass in Go: the equality oracle for the WGSL kernels
// and the native fallback. It holds every weight in memory.
//
// ExactGGML reproduces llama.cpp's CPU backend: activations are quantised to
// Q8_0 (Q8_1 for Q4_1 weights) before every quantised matrix product and the
// KV cache is stored in binary16. Without it the products use f32
// activations and an f32 cache, which is what the GPU kernels compute.
type CPU struct {
	Cfg       Config
	ExactGGML bool
	// F16KV alone stores the KV cache in binary16 (what llama.cpp's default
	// cache type does) without quantising activations.
	F16KV  bool
	w      map[string]*cpuTensor
	k, v   [][]float32 // per layer: ctx * kvHeads * headDim
	pos    int
	maxCtx int
	// MaxLayers, when non-zero, runs only that many layers and returns nil
	// (debugging); LastX is the residual stream after the layers run.
	MaxLayers int
	LastX     []float32
}

type cpuTensor struct {
	t    *gguf.Tensor
	data []byte
}

// LoadCPU reads every tensor of f from r.
func LoadCPU(f *gguf.File, r io.ReaderAt, maxCtx int) (*CPU, error) {
	cfg, err := ConfigFrom(f)
	if err != nil {
		return nil, err
	}
	m := &CPU{Cfg: cfg, w: map[string]*cpuTensor{}, maxCtx: maxCtx}
	for i := range f.Tensors {
		t := &f.Tensors[i]
		b := make([]byte, t.Size)
		if _, err := r.ReadAt(b, int64(f.DataOffset+t.Offset)); err != nil {
			return nil, fmt.Errorf("llm: tensor %s: %w", t.Name, err)
		}
		m.w[t.Name] = &cpuTensor{t, b}
	}
	kv := maxCtx * cfg.KVHeads * cfg.HeadDim
	for l := 0; l < cfg.Layers; l++ {
		m.k = append(m.k, make([]float32, kv))
		m.v = append(m.v, make([]float32, kv))
	}
	return m, nil
}

// Reset clears the KV cache.
func (m *CPU) Reset() { m.pos = 0 }

// Pos is the number of tokens in the cache.
func (m *CPU) Pos() int { return m.pos }

func (m *CPU) tensor(name string) *cpuTensor {
	t, ok := m.w[name]
	if !ok {
		panic("llm: missing tensor " + name)
	}
	return t
}

// rmsF64 sums the squares in float64: a perturbation of about 1e-7 relative
// (tests of how an activation-quantised path amplifies rounding differences).
var rmsF64 bool

func rmsNorm(dst, x, w []float32, eps float32) {
	var ss float32
	if rmsF64 {
		var s float64
		for _, v := range x {
			s += float64(v) * float64(v)
		}
		ss = float32(s)
	} else {
		for _, v := range x {
			ss += v * v
		}
	}
	s := float32(1 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i, v := range x {
		dst[i] = v * s * w[i]
	}
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
	}
	return out
}

func u16(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }

// q8 is a quantised activation row (ggml block_q8_0 / block_q8_1).
type q8 struct {
	d  []float32 // f16-rounded scales
	s  []float32 // f16-rounded d*sum(qs) (Q8_1)
	qs []int8
}

func quantizeQ8(x []float32) q8 {
	nb := len(x) / 32
	y := q8{d: make([]float32, nb), s: make([]float32, nb), qs: make([]int8, len(x))}
	for i := 0; i < nb; i++ {
		var amax float32
		for _, v := range x[i*32 : i*32+32] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := amax / 127
		var id float32
		if d != 0 {
			id = 1 / d
		}
		y.d[i] = RoundF16(d)
		sum := 0
		for j := 0; j < 32; j++ {
			q := int8(math.Round(float64(x[i*32+j] * id)))
			y.qs[i*32+j] = q
			sum += int(q)
		}
		y.s[i] = RoundF16(d * float32(sum))
	}
	return y
}

// matvec computes dst = W x for a [in, out] tensor.
func (m *CPU) matvec(dst, x []float32, t *cpuTensor, qx *q8) {
	in, out := int(t.t.Dims[0]), int(t.t.Dims[1])
	_, bb, _ := t.t.Type.BlockSize()
	rowBytes := in / 32 * int(bb)
	if t.t.Type == gguf.TypeF32 {
		w := f32s(t.data)
		for o := 0; o < out; o++ {
			var s float32
			for i := 0; i < in; i++ {
				s += w[o*in+i] * x[i]
			}
			dst[o] = s
		}
		return
	}
	for o := 0; o < out; o++ {
		row := t.data[o*rowBytes : (o+1)*rowBytes]
		var sum float32
		for b := 0; b < in/32; b++ {
			blk := row[b*int(bb) : (b+1)*int(bb)]
			switch t.t.Type {
			case gguf.TypeQ4_0:
				d := F16ToF32(u16(blk))
				if qx != nil {
					si := 0
					for j := 0; j < 16; j++ {
						si += (int(blk[2+j]&15) - 8) * int(qx.qs[b*32+j])
						si += (int(blk[2+j]>>4) - 8) * int(qx.qs[b*32+j+16])
					}
					sum += d * qx.d[b] * float32(si)
				} else {
					var s float32
					for j := 0; j < 16; j++ {
						s += float32(int(blk[2+j]&15)-8) * x[b*32+j]
						s += float32(int(blk[2+j]>>4)-8) * x[b*32+j+16]
					}
					sum += d * s
				}
			case gguf.TypeQ4_1:
				d, mn := F16ToF32(u16(blk)), F16ToF32(u16(blk[2:]))
				if qx != nil {
					si := 0
					for j := 0; j < 16; j++ {
						si += int(blk[4+j]&15) * int(qx.qs[b*32+j])
						si += int(blk[4+j]>>4) * int(qx.qs[b*32+j+16])
					}
					sum += d*qx.d[b]*float32(si) + mn*qx.s[b]
				} else {
					var s, sx float32
					for j := 0; j < 16; j++ {
						s += float32(blk[4+j]&15) * x[b*32+j]
						s += float32(blk[4+j]>>4) * x[b*32+j+16]
					}
					for j := 0; j < 32; j++ {
						sx += x[b*32+j]
					}
					sum += d*s + mn*sx
				}
			case gguf.TypeQ8_0:
				d := F16ToF32(u16(blk))
				if qx != nil {
					si := 0
					for j := 0; j < 32; j++ {
						si += int(int8(blk[2+j])) * int(qx.qs[b*32+j])
					}
					sum += d * qx.d[b] * float32(si)
				} else {
					var s float32
					for j := 0; j < 32; j++ {
						s += float32(int8(blk[2+j])) * x[b*32+j]
					}
					sum += d * s
				}
			default:
				panic(fmt.Sprintf("llm: CPU matvec: ggml type %d", t.t.Type))
			}
		}
		dst[o] = sum
	}
}

// Forward runs one token at the next position and returns the logits.
func (m *CPU) Forward(tok int32) []float32 {
	c := m.Cfg
	if m.pos >= m.maxCtx {
		panic("llm: context full")
	}
	x := make([]float32, c.Embd)
	emb := m.tensor("token_embd.weight")
	rowBytes := c.Embd / 32 * 34
	row := emb.data[int(tok)*rowBytes : (int(tok)+1)*rowBytes]
	for b := 0; b < c.Embd/32; b++ { // dequantise one Q8_0 row
		d := F16ToF32(u16(row[b*34:]))
		for j := 0; j < 32; j++ {
			x[b*32+j] = d * float32(int8(row[b*34+2+j]))
		}
	}
	h := make([]float32, c.Embd)
	q := make([]float32, c.Heads*c.HeadDim)
	kk := make([]float32, c.KVHeads*c.HeadDim)
	vv := make([]float32, c.KVHeads*c.HeadDim)
	att := make([]float32, c.Heads*c.HeadDim)
	o := make([]float32, c.Embd)
	g := make([]float32, c.FF)
	u := make([]float32, c.FF)
	dn := make([]float32, c.Embd)
	scores := make([]float32, m.pos+1)
	kvd := c.KVHeads * c.HeadDim
	prep := func(v []float32) *q8 {
		if !m.ExactGGML {
			return nil
		}
		y := quantizeQ8(v)
		return &y
	}
	nl := c.Layers
	if m.MaxLayers != 0 {
		nl = min(nl, max(m.MaxLayers, 0))
	}
	for l := 0; l < nl; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		rmsNorm(h, x, f32s(m.tensor(p+"attn_norm.weight").data), c.Eps)
		qh := prep(h)
		m.matvec(q, h, m.tensor(p+"attn_q.weight"), qh)
		m.matvec(kk, h, m.tensor(p+"attn_k.weight"), qh)
		m.matvec(vv, h, m.tensor(p+"attn_v.weight"), qh)
		rope(q, m.pos, c.Heads, c.HeadDim, c.RopeBase)
		rope(kk, m.pos, c.KVHeads, c.HeadDim, c.RopeBase)
		for i := 0; i < kvd; i++ {
			if m.ExactGGML || m.F16KV {
				m.k[l][m.pos*kvd+i], m.v[l][m.pos*kvd+i] = RoundF16(kk[i]), RoundF16(vv[i])
			} else {
				m.k[l][m.pos*kvd+i], m.v[l][m.pos*kvd+i] = kk[i], vv[i]
			}
		}
		scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
		grp := c.Heads / c.KVHeads
		for hd := 0; hd < c.Heads; hd++ {
			kvh := hd / grp
			qv := q[hd*c.HeadDim : (hd+1)*c.HeadDim]
			mx := float32(math.Inf(-1))
			for t := 0; t <= m.pos; t++ {
				kr := m.k[l][t*kvd+kvh*c.HeadDim:]
				var s float32
				for i := 0; i < c.HeadDim; i++ {
					s += qv[i] * kr[i]
				}
				scores[t] = s * scale
				mx = max(mx, scores[t])
			}
			var sum float32
			for t := 0; t <= m.pos; t++ {
				scores[t] = float32(math.Exp(float64(scores[t] - mx)))
				sum += scores[t]
			}
			out := att[hd*c.HeadDim : (hd+1)*c.HeadDim]
			for i := range out {
				out[i] = 0
			}
			for t := 0; t <= m.pos; t++ {
				vr := m.v[l][t*kvd+kvh*c.HeadDim:]
				w := scores[t] / sum
				for i := range out {
					out[i] += w * vr[i]
				}
			}
		}
		m.matvec(o, att, m.tensor(p+"attn_output.weight"), prep(att))
		for i := range x {
			x[i] += o[i]
		}
		rmsNorm(h, x, f32s(m.tensor(p+"ffn_norm.weight").data), c.Eps)
		qh = prep(h)
		m.matvec(g, h, m.tensor(p+"ffn_gate.weight"), qh)
		m.matvec(u, h, m.tensor(p+"ffn_up.weight"), qh)
		for i := range g {
			g[i] = g[i] / (1 + float32(math.Exp(float64(-g[i])))) * u[i]
		}
		m.matvec(dn, g, m.tensor(p+"ffn_down.weight"), prep(g))
		for i := range x {
			x[i] += dn[i]
		}
	}
	m.LastX = slices.Clone(x)
	if m.MaxLayers != 0 && m.MaxLayers < c.Layers {
		m.pos++
		return nil
	}
	rmsNorm(h, x, f32s(m.tensor("output_norm.weight").data), c.Eps)
	logits := make([]float32, c.Vocab)
	m.matvec(logits, h, emb, prep(h)) // tied output embedding
	m.pos++
	return logits
}

// rope rotates adjacent pairs (llama.cpp rope type NORM): pair i of a head
// turns by pos * base^(-2i/dim).
func rope(x []float32, pos, heads, dim int, base float32) {
	for h := 0; h < heads; h++ {
		for i := 0; i < dim/2; i++ {
			theta := float64(pos) * math.Pow(float64(base), -float64(2*i)/float64(dim))
			s, c := math.Sincos(theta)
			a, b := x[h*dim+2*i], x[h*dim+2*i+1]
			x[h*dim+2*i] = a*float32(c) - b*float32(s)
			x[h*dim+2*i+1] = a*float32(s) + b*float32(c)
		}
	}
}

// Argmax returns the index of the largest logit (the first on ties, as
// llama.cpp's greedy sampler does).
func Argmax(l []float32) int32 {
	best := 0
	for i, v := range l {
		if v > l[best] {
			best = i
		}
	}
	return int32(best)
}
