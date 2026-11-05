package llm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/llm/gguf"
)

// TPR is the number of invocations that share one output row of the fused
// matrix-vector kernels (a power of two dividing the workgroup size; the
// workgroup then computes WG/TPR rows). It trades the reduction depth against
// the rows per workgroup.
var TPR = 8

// fusedGrid is a dispatch size.
type fusedGrid struct{ x, y uint32 }

// fusedBG is the five bind groups of a fused layer.
type fusedBG struct {
	k1, k2, k3, k4, k5 *webgpu.BindGroup
	k5n                string
}

// fusedOK says whether the fused kernels apply to the model on this adapter.
func (g *GPU) fusedOK() bool {
	c := g.Cfg
	kvd := c.KVHeads * c.HeadDim
	rpw := g.wg / TPR
	return TPR >= 2 && g.wg%TPR == 0 && rpw >= 2 && rpw%2 == 0 &&
		c.HeadDim%4 == 0 && c.KVHeads > 0 && c.Heads%c.KVHeads == 0 && g.wg%(c.HeadDim/4) == 0 && c.HeadDim/4 <= g.wg &&
		c.Embd%32 == 0 && c.FF%32 == 0 && kvd%4 == 0 && c.HeadDim%2 == 0
}

// layerFusable: the six matrices of the attention and gate/up are Q4_0, the
// down matrix Q4_0 or Q4_1 (the mix a Q4_0 file made by llama-quantize has).
func layerFusable(f *gguf.File, p string) bool {
	for _, n := range []string{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up"} {
		t, ok := f.Tensor(p + n + ".weight")
		if !ok || t.Type != gguf.TypeQ4_0 {
			return false
		}
	}
	t, ok := f.Tensor(p + "ffn_down.weight")
	return ok && (t.Type == gguf.TypeQ4_0 || t.Type == gguf.TypeQ4_1)
}

func (g *GPU) fusedKernel(name string, s mvSpec, k mvKind, extra map[string]int) error {
	if _, ok := g.k[name]; ok {
		return nil
	}
	src, access, n := fusedMVSource(s, k, extra)
	if err := g.newKernel(name, src, access); err != nil {
		return err
	}
	x := min(n, int(g.lim.MaxWorkgroupsPerDimension))
	g.fgrid[name] = fusedGrid{uint32(x), uint32((n + x - 1) / x)}
	return nil
}

// chunks is the number of attention chunks (WG positions each) of a full context.
func (g *GPU) chunks() int { return (g.ctx + g.wg - 1) / g.wg }

func (g *GPU) spec(n, rows int) mvSpec {
	return mvSpec{wg: g.wg, tpr: TPR, maxShared: int(g.lim.MaxWorkgroupStorage), n: n, rows: rows, maxGroups: int(g.lim.MaxWorkgroupsPerDimension), eps: g.Cfg.Eps}
}

// fusedKernels creates the pipelines the fused layers use (once).
func (g *GPU) fusedKernels(down41 bool) error {
	c := g.Cfg
	kvd := c.KVHeads * c.HeadDim
	nqkv := c.Embd + 2*kvd
	if err := g.fusedKernel("k1", g.spec(c.Embd, nqkv), mvKind{mode: "qkv", norm: true}, map[string]int{
		"NQ": c.Embd, "NK": kvd, "HD": c.HeadDim, "HP": c.HeadDim / 2, "KVD": kvd, "CTXKVD": g.ctx * kvd, "NW": nqkv * c.Embd / 32,
		"BIAS": b2i(c.Bias), "BO": nqkv*c.Embd/32 + c.Embd}); err != nil {
		return err
	}
	if _, ok := g.k["k2"]; !ok {
		if err := g.newKernel("k2", fusedAttnSource(g.wg, c.HeadDim, c.Heads, c.KVHeads, g.ctx, float32(1/math.Sqrt(float64(c.HeadDim)))), "rrwwr"); err != nil {
			return err
		}
	}
	if err := g.fusedKernel("k3", g.spec(c.Embd, c.Embd), mvKind{mode: "acc", attnIn: true}, map[string]int{"HD4": c.HeadDim / 4, "S": g.chunks()}); err != nil {
		return err
	}
	if err := g.fusedKernel("k4", g.spec(c.Embd, 2*c.FF), mvKind{mode: "gateup", norm: true}, map[string]int{"F": c.FF, "NW": 2 * c.FF * c.Embd / 32}); err != nil {
		return err
	}
	kn := "k5_0"
	if down41 {
		kn = "k5_1"
	}
	return g.fusedKernel(kn, g.spec(c.FF, c.Embd), mvKind{mode: "acc", q41: down41}, nil)
}

// loadGroup streams matrices of one type, one after the other, into a single
// quant buffer and a scale buffer; tail floats (a norm's weights) follow the
// scales in the scale buffer.
//
// biases (F32 tensors, one per matrix, may be nil) follow the norm weights.
// perm marks the matrices whose rows are reordered inside each head of hd rows
// from the NEOX layout (element i and i+hd/2 are a rotary pair) to adjacent
// pairs (2i, 2i+1): new row 2i+s is old row i+s*hd/2, and the biases likewise.
// The scores q.k are unchanged by a permutation applied to both Q and K.
func (g *GPU) loadGroup(u *uploader, f *gguf.File, r io.ReaderAt, names []string, norm string, biases []string, perm []bool, hd int) (qs, aux *webgpu.Buffer, err error) {
	var ts []gguf.Tensor
	var nblk uint64
	for _, n := range names {
		t, ok := f.Tensor(n)
		if !ok {
			return nil, nil, fmt.Errorf("llm: missing tensor %s", n)
		}
		_, bb, _ := t.Type.BlockSize()
		nblk += t.Size / bb
		ts = append(ts, *t)
	}
	tail := uint64(0)
	var nw gguf.Tensor
	if norm != "" {
		t, ok := f.Tensor(norm)
		if !ok || t.Type != gguf.TypeF32 {
			return nil, nil, fmt.Errorf("llm: tensor %s missing or not F32", norm)
		}
		nw, tail = *t, t.Size
	}
	var bts []gguf.Tensor
	for _, n := range biases {
		if n == "" {
			continue
		}
		t, ok := f.Tensor(n)
		if !ok || t.Type != gguf.TypeF32 {
			return nil, nil, fmt.Errorf("llm: tensor %s missing or not F32", n)
		}
		bts = append(bts, *t)
		tail += t.Size
	}
	if qs, err = g.buffer(nblk*16, stor); err != nil {
		return nil, nil, err
	}
	if aux, err = g.buffer(nblk*4+tail, stor); err != nil {
		return nil, nil, err
	}
	off := uint64(0)
	for i, t := range ts {
		if perm != nil && perm[i] {
			if err := g.streamMatNeox(u, f, r, t, qs, aux, off, hd); err != nil {
				return nil, nil, err
			}
		} else if err := g.streamMat(u, f, r, t, qs, aux, nil, off); err != nil {
			return nil, nil, err
		}
		_, bb, _ := t.Type.BlockSize()
		off += t.Size / bb
	}
	at := nblk * 4
	if norm != "" {
		data := make([]byte, nw.Size)
		if _, err := r.ReadAt(data, int64(f.DataOffset+nw.Offset)); err != nil {
			return nil, nil, err
		}
		if err := u.write(aux, at, data); err != nil {
			return nil, nil, err
		}
		at += nw.Size
	}
	bi := 0
	for i, n := range biases {
		if n == "" {
			continue
		}
		bt := bts[bi]
		bi++
		data := make([]byte, bt.Size)
		if _, err := r.ReadAt(data, int64(f.DataOffset+bt.Offset)); err != nil {
			return nil, nil, err
		}
		if perm != nil && perm[i] {
			data = neoxRows(data, 4, hd)
		}
		if err := u.write(aux, at, data); err != nil {
			return nil, nil, err
		}
		at += bt.Size
	}
	return qs, aux, nil
}

// fusedLayer loads one layer's weights in the fused form and makes its bind groups.
func (g *GPU) fusedLayer(u *uploader, f *gguf.File, r io.ReaderAt, p string, x, q, act, pv, ml, state, cs *webgpu.Buffer) (*fusedBG, error) {
	c := g.Cfg
	kvd := c.KVHeads * c.HeadDim
	var fb fusedBG
	var biases []string
	if c.Bias {
		biases = []string{p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias"}
	}
	var perm []bool
	if c.RopeNeox {
		perm = []bool{true, true, false}
	}
	qkvQS, qkvAux, err := g.loadGroup(u, f, r, []string{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, p+"attn_norm.weight", biases, perm, c.HeadDim)
	if err != nil {
		return nil, err
	}
	guQS, guAux, err := g.loadGroup(u, f, r, []string{p + "ffn_gate.weight", p + "ffn_up.weight"}, p+"ffn_norm.weight", nil, nil, 0)
	if err != nil {
		return nil, err
	}
	o, err := g.loadMat(u, f, r, p+"attn_output.weight")
	if err != nil {
		return nil, err
	}
	down, err := g.loadMat(u, f, r, p+"ffn_down.weight")
	if err != nil {
		return nil, err
	}
	kv, err := g.buffer(uint64(4*2*g.ctx*kvd), stor)
	if err != nil {
		return nil, err
	}
	g.named["kv"+p] = kv
	if fb.k1, err = g.bind("k1", qkvQS, qkvAux, x, q, kv, state, cs); err != nil {
		return nil, err
	}
	if fb.k2, err = g.bind("k2", q, kv, pv, ml, state); err != nil {
		return nil, err
	}
	if fb.k3, err = g.bind("k3", o.qs, o.d, pv, ml, state, x); err != nil {
		return nil, err
	}
	if fb.k4, err = g.bind("k4", guQS, guAux, x, act); err != nil {
		return nil, err
	}
	if down.typ == gguf.TypeQ4_1 {
		fb.k5n = "k5_1"
		fb.k5, err = g.bind("k5_1", down.qs, down.d, down.m, act, x)
	} else {
		fb.k5n = "k5_0"
		fb.k5, err = g.bind("k5_0", down.qs, down.d, act, x)
	}
	if err != nil {
		return nil, err
	}
	return &fb, nil
}

func (g *GPU) dispatchGrid(p *webgpu.ComputePassRecorder, kn string, bg *webgpu.BindGroup, m int) {
	gr := g.fgrid[kn]
	g.dispatch3(p, kn, bg, gr.x, gr.y, uint32(m))
}

// recordFused records the five dispatches of a fused layer for m tokens at
// the positions g.pos, g.pos+1, ... (the z axis of every dispatch is the
// token). The attention of token t sees positions up to g.pos+t.
func (g *GPU) recordFused(p *webgpu.ComputePassRecorder, l *fusedBG, m int) {
	g.dispatchGrid(p, "k1", l.k1, m)
	g.dispatch3(p, "k2", l.k2, uint32((g.pos+m+g.wg-1)/g.wg), uint32(g.Cfg.Heads), uint32(m))
	g.dispatchGrid(p, "k3", l.k3, m)
	g.dispatchGrid(p, "k4", l.k4, m)
	g.dispatchGrid(p, l.k5n, l.k5, m)
}

// recordHead records the final norm, the logits and the sampling of the token
// whose residual stream is x[0:Embd].
func (g *GPU) recordHead(p *webgpu.ComputePassRecorder, fromLast bool) {
	if fromLast {
		g.dispatch(p, "rms", g.rmsL, 1, 1)
	} else {
		g.dispatch(p, "rms", g.rmsF, 1, 1)
	}
	for _, hd := range g.heads {
		if hd.fused {
			g.dispatchGrid(p, hd.kn, hd.bg, 1)
			continue
		}
		x, y := g.grid(hd.rows)
		g.dispatch(p, hd.kn, hd.bg, x, y)
	}
	if g.am1 != nil {
		g.dispatch(p, "am1", g.am1, argmaxGroups, 1)
		g.dispatch(p, "am2", g.am2, 1, 1)
	} else {
		g.dispatch(p, "argmax", g.argmax, 1, 1)
	}
}

// chunk records m prompt tokens through the fused layers; with last, the
// final token's logits and its sampled token too (a second pass: the
// residual stream of the last token is first copied to the front of x).
func (g *GPU) chunk(rec *webgpu.Recorder, m int, last bool) error {
	adv := m
	if last {
		adv = m - 1
	}
	rec.WriteBuffer(g.advNA, 0, le32(uint32(adv)))
	p := rec.BeginComputePass()
	g.dispatch3(p, g.embedK, g.embed, 1, 1, uint32(m))
	for _, l := range g.layers {
		g.recordFused(p, l.fx, m)
	}
	if adv > 0 {
		g.dispatch(p, "advn", g.advA, 1, 1)
	}
	if err := p.End(); err != nil {
		return err
	}
	g.pos += m
	if !last {
		return nil
	}
	rec.CopyBufferToBuffer(g.xbuf, uint64(4*(m-1)*g.Cfg.Embd), g.xlast, 0, uint64(4*g.Cfg.Embd))
	p = rec.BeginComputePass()
	g.recordHead(p, true)
	g.dispatch(p, "advn", g.advB, 1, 1)
	return p.End()
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// neoxMap is the row of the NEOX layout that goes to row nr of the adjacent-pair
// layout, in heads of hd rows: row 2i+s of a head is row i+s*hd/2.
func neoxMap(nr, hd int) int {
	h, k := nr/hd, nr%hd
	return h*hd + k/2 + (k%2)*(hd/2)
}

// neoxRows reorders rows of rowBytes bytes (see loadGroup).
func neoxRows(data []byte, rowBytes, hd int) []byte {
	out := make([]byte, len(data))
	n := len(data) / rowBytes
	for nr := 0; nr < n; nr++ {
		copy(out[nr*rowBytes:(nr+1)*rowBytes], data[neoxMap(nr, hd)*rowBytes:])
	}
	return out
}

// streamMatNeox is streamMat for a Q4_0 matrix of whole rows whose rows are
// reordered by neoxMap: the tensor is read whole (a Q or K projection is a few
// MiB) and its blocks written in the new order.
func (g *GPU) streamMatNeox(u *uploader, f *gguf.File, r io.ReaderAt, t gguf.Tensor, qsBuf, dBuf *webgpu.Buffer, blkOff uint64, hd int) error {
	if t.Type != gguf.TypeQ4_0 || len(t.Dims) != 2 {
		return fmt.Errorf("llm: tensor %s: the NEOX row order is implemented for Q4_0 matrices", t.Name)
	}
	rows, nbr := int(t.Dims[1]), int(t.Dims[0])/32
	if hd <= 0 || hd%2 != 0 || rows%hd != 0 {
		return fmt.Errorf("llm: tensor %s: %d rows is not a whole number of heads of %d", t.Name, rows, hd)
	}
	raw := make([]byte, t.Size)
	if _, err := r.ReadAt(raw, int64(f.DataOffset+t.Offset)); err != nil {
		return fmt.Errorf("llm: reading %s: %w", t.Name, err)
	}
	qsb := make([]byte, rows*nbr*16)
	db := make([]byte, rows*nbr*4)
	for nr := 0; nr < rows; nr++ {
		or := neoxMap(nr, hd)
		for b := 0; b < nbr; b++ {
			blk := raw[(or*nbr+b)*18 : (or*nbr+b+1)*18]
			binary.LittleEndian.PutUint32(db[4*(nr*nbr+b):], math.Float32bits(F16ToF32(u16(blk))))
			copy(qsb[(nr*nbr+b)*16:], blk[2:18])
		}
	}
	if err := u.write(qsBuf, blkOff*16, qsb); err != nil {
		return err
	}
	return u.write(dBuf, blkOff*4, db)
}
