package llm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/llm/gguf"
)

// ChunkSize is the size of one weight upload: tensors are read from the
// model file and sent to the GPU in pieces of at most this many bytes, so the
// whole model never sits in guest memory.
const ChunkSize = 16 << 20

// PieceSize is the size of one read from the model file while loading.
const PieceSize = 4 << 20

// Limits are the adapter limits the kernels adapt to (the capset's
// DeviceLimits, go-virtio/gpu wgpubatch). Nothing in the kernels assumes a
// vendor's values: the workgroup size and the grids are derived from these.
type Limits struct {
	MaxInvocationsPerWorkgroup uint32
	MaxWorkgroupStorage        uint32 // bytes
	MaxWorkgroupsPerDimension  uint32
	MaxStorageBuffers          uint32 // per shader stage
	MaxBindingSize             uint64 // maxStorageBufferBindingSize
	MaxBufferSize              uint64
	MemoryBudget               uint64 // GPU memory the container may hold (capset max_memory)
}

// DefaultLimits are WebGPU's defaults (the guaranteed minimum).
func DefaultLimits() Limits {
	return Limits{256, 16384, 65535, 8, 128 << 20, 256 << 20, 256 << 20}
}

// WorkgroupSize is the workgroup size of the reductions: 64 (a size every
// WebGPU adapter supports, a multiple of the 32-wide and 64-wide waves of
// current hardware but not relying on either), reduced when the adapter's
// invocation or workgroup-memory limit is lower. A power of two.
func (l Limits) WorkgroupSize() int {
	wg := 64
	for wg > 1 && (uint32(wg) > l.MaxInvocationsPerWorkgroup || uint32(wg*8) > l.MaxWorkgroupStorage) {
		wg /= 2
	}
	return wg
}

// Stats counts what a run did.
type Stats struct {
	Submits    int
	Dispatches int
	Tokens     int
}

type kernel struct {
	pipe *webgpu.ComputePipeline
	bgl  *webgpu.BindGroupLayout
}

type mat struct {
	typ      gguf.Type
	in, out  int
	qs, d, m *webgpu.Buffer
	bytes    uint64
}

type layerBG struct {
	rms1, q, k, v, rope, attn, o, rms2, gate, up, swi, down *webgpu.BindGroup
	fx                                                      *fusedBG // the fused form (fused.go), when set the fields above are not
}

// GPU is the forward pass on the GPU for the llama architecture, in f32.
type GPU struct {
	dev  *webgpu.Device
	lim  Limits
	Cfg  Config
	ctx  int
	wg   int
	used uint64

	k     map[string]*kernel
	kinds map[*webgpu.BindGroup]string

	state, tokens, outb, outRead, logits, logitsRead *webgpu.Buffer

	layers []layerBG
	embed  *webgpu.BindGroup
	rmsF   *webgpu.BindGroup
	logitB *webgpu.BindGroup
	argmax *webgpu.BindGroup
	adv    *webgpu.BindGroup

	rec   *webgpu.Recorder
	Stats Stats

	raw, qsb, db, mb []byte // load scratch, reused
	named            map[string]*webgpu.Buffer
	keep             []any // every GPU object, alive for the life of the model (remote objects are released when collected)
	// Layers, when non-zero, runs only that many layers (debugging; no
	// logits and no sampling are then produced).
	Layers int
	Skip   map[string]bool
	fgrid  map[string]fusedGrid
	pos    int // the position of the next token, as the state buffer holds it on the GPU
	pv, ml *webgpu.Buffer
	am1    *webgpu.BindGroup // two-stage argmax (fused.go), nil with NoFused
	am2    *webgpu.BindGroup

	pm         int // tokens per prompt chunk (1: the token-at-a-time path)
	xlast      *webgpu.Buffer
	rmsL       *webgpu.BindGroup
	xbuf       *webgpu.Buffer    // the residual stream, pm vectors
	advA, advB *webgpu.BindGroup // state += n (n from advNA), state += 1
	advNA      *webgpu.Buffer
}

// PrefillBatch is the number of prompt tokens the fused layers process
// together in one pass: every weight matrix is read once per chunk and the
// dispatches are shared. A chunk of 64 tokens takes about 12 ms of GPU time
// on an Apple M4 Max, far below the 500 ms budget of a submit.
var PrefillBatch = 64

// NoFused turns the fused layer kernels off (the v0.2.0 path, for A/B runs).
var NoFused bool

func (g *GPU) buffer(size uint64, usage webgpu.BufferUsage) (*webgpu.Buffer, error) {
	size = (size + 3) &^ 3
	if size > g.lim.MaxBindingSize && usage&webgpu.BufferUsageStorage != 0 {
		return nil, fmt.Errorf("llm: a %d-byte buffer exceeds the adapter's maxStorageBufferBindingSize %d (this model needs weights split across buffers)", size, g.lim.MaxBindingSize)
	}
	if g.used+size > g.lim.MemoryBudget {
		return nil, fmt.Errorf("llm: the model needs more GPU memory than the %d bytes granted (run with a larger --gpu-memory)", g.lim.MemoryBudget)
	}
	b, err := g.dev.CreateBuffer(webgpu.BufferDescriptor{Size: size, Usage: usage})
	if err != nil {
		return nil, err
	}
	g.used += size
	g.keep = append(g.keep, b) // a bind group holds only the id: the Buffer must outlive it
	return b, nil
}

const stor = webgpu.BufferUsageStorage | webgpu.BufferUsageCopyDst | webgpu.BufferUsageCopySrc

func (g *GPU) newKernel(name, src, access string) error {
	m, err := g.dev.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: wgsl(src, g.wg)})
	if err != nil {
		return fmt.Errorf("llm: kernel %s: %w", name, err)
	}
	var es []webgpu.BindGroupLayoutEntry
	for i, c := range access {
		t := webgpu.BufferBindingTypeReadOnlyStorage
		if c == 'w' {
			t = webgpu.BufferBindingTypeStorage
		}
		es = append(es, webgpu.BindGroupLayoutEntry{Binding: uint32(i), Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: t}})
	}
	if uint32(len(es)) > g.lim.MaxStorageBuffers {
		return fmt.Errorf("llm: kernel %s needs %d storage buffers, the adapter allows %d", name, len(es), g.lim.MaxStorageBuffers)
	}
	bgl, err := g.dev.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: es})
	if err != nil {
		return err
	}
	pl, err := g.dev.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{bgl}})
	if err != nil {
		return err
	}
	p, err := g.dev.CreateComputePipeline(webgpu.ComputePipelineDescriptor{Layout: pl, Compute: webgpu.ProgrammableStage{Module: m, EntryPoint: "main"}})
	if err != nil {
		return err
	}
	g.k[name] = &kernel{p, bgl}
	g.keep = append(g.keep, m, pl, p, bgl)
	return nil
}

func (g *GPU) bind(kn string, bufs ...*webgpu.Buffer) (*webgpu.BindGroup, error) {
	var es []webgpu.BindGroupEntry
	for i, b := range bufs {
		es = append(es, webgpu.BindGroupEntry{Binding: uint32(i), Buffer: b})
	}
	return g.dev.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: g.k[kn].bgl, Entries: es})
}

// uploader batches small writes and sends big ones in ChunkSize pieces.
type uploader struct {
	g       *GPU
	rec     *webgpu.Recorder
	pending int
}

func (u *uploader) write(b *webgpu.Buffer, off uint64, data []byte) error {
	for len(data) > 0 {
		n := min(len(data), ChunkSize)
		if u.pending > 0 && u.pending+n > ChunkSize {
			if err := u.flush(); err != nil {
				return err
			}
		}
		if u.rec == nil {
			u.rec = u.g.dev.NewRecorder()
		}
		u.rec.WriteBuffer(b, off, data[:n])
		u.pending += n
		off += uint64(n)
		data = data[n:]
	}
	return nil
}

func (u *uploader) flush() error {
	if u.rec == nil {
		return nil
	}
	if _, err := u.rec.Submit(); err != nil {
		return err
	}
	u.g.Stats.Submits++
	u.rec, u.pending = nil, 0
	// The recorder and the sink's frame held two copies of the chunk; give
	// them back now, so the guest's memory stays a few chunks, not the model.
	runtime.GC()
	return nil
}

func le32(vals ...uint32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	return b
}

func (g *GPU) metaBuf(u *uploader, vals ...uint32) (*webgpu.Buffer, error) {
	b, err := g.buffer(uint64(4*len(vals)), stor)
	if err != nil {
		return nil, err
	}
	return b, u.write(b, 0, le32(vals...))
}

// loadMat streams one quantised matrix to the GPU: the 18-, 20- or 34-byte
// blocks of the file become aligned arrays (packed quants as u32 words,
// scales and minima as f32, exact from binary16).
func (g *GPU) loadMat(u *uploader, f *gguf.File, r io.ReaderAt, name string) (*mat, error) {
	t, ok := f.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("llm: missing tensor %s", name)
	}
	if len(t.Dims) != 2 {
		return nil, fmt.Errorf("llm: tensor %s is not a matrix", name)
	}
	m := &mat{typ: t.Type, in: int(t.Dims[0]), out: int(t.Dims[1])}
	_, bb, _ := t.Type.BlockSize()
	var qb uint64 // quant bytes per block
	var hasM bool
	switch t.Type {
	case gguf.TypeQ4_0:
		qb = 16
	case gguf.TypeQ4_1:
		qb, hasM = 16, true
	case gguf.TypeQ8_0:
		qb = 32
	default:
		return nil, fmt.Errorf("llm: tensor %s has ggml type %d, which has no GPU kernel yet (Q4_0, Q4_1, Q8_0)", name, t.Type)
	}
	nblk := t.Size / bb
	var err error
	if m.qs, err = g.buffer(nblk*qb, stor); err != nil {
		return nil, err
	}
	if m.d, err = g.buffer(nblk*4, stor); err != nil {
		return nil, err
	}
	if hasM {
		if m.m, err = g.buffer(nblk*4, stor); err != nil {
			return nil, err
		}
	}
	return m, g.streamMat(u, f, r, *t, m.qs, m.d, m.m, 0)
}

// streamMat writes the blocks of tensor t into the buffers qs, d (and mm for
// Q4_1) from block blkOff on: a matrix alone (0) or one of a concatenated
// group.
func (g *GPU) streamMat(u *uploader, f *gguf.File, r io.ReaderAt, t gguf.Tensor, qsBuf, dBuf, mBuf *webgpu.Buffer, blkOff uint64) error {
	name := t.Name
	_, bb, _ := t.Type.BlockSize()
	qb := uint64(16)
	if t.Type == gguf.TypeQ8_0 {
		qb = 32
	}
	hasM := t.Type == gguf.TypeQ4_1
	nblk := t.Size / bb
	// Pieces of PieceSize raw bytes, in scratch buffers allocated once: the
	// guest's own memory stays a few times PieceSize however big the model.
	per := uint64(PieceSize) / bb
	if g.raw == nil {
		g.raw, g.qsb, g.db, g.mb = make([]byte, PieceSize), make([]byte, PieceSize), make([]byte, PieceSize/4), make([]byte, PieceSize/4)
	}
	raw, qsb, db, mb := g.raw, g.qsb, g.db, g.mb
	for b0 := uint64(0); b0 < nblk; b0 += per {
		n := min(per, nblk-b0)
		if _, err := r.ReadAt(raw[:n*bb], int64(f.DataOffset+t.Offset+b0*bb)); err != nil {
			return fmt.Errorf("llm: reading %s: %w", name, err)
		}
		for i := uint64(0); i < n; i++ {
			blk := raw[i*bb : (i+1)*bb]
			binary.LittleEndian.PutUint32(db[4*i:], math.Float32bits(F16ToF32(u16(blk))))
			off := uint64(2)
			if hasM {
				binary.LittleEndian.PutUint32(mb[4*i:], math.Float32bits(F16ToF32(u16(blk[2:]))))
				off = 4
			}
			copy(qsb[i*qb:], blk[off:off+qb])
		}
		if err := u.write(qsBuf, (blkOff+b0)*qb, qsb[:n*qb]); err != nil {
			return err
		}
		if err := u.write(dBuf, (blkOff+b0)*4, db[:n*4]); err != nil {
			return err
		}
		if hasM {
			if err := u.write(mBuf, (blkOff+b0)*4, mb[:n*4]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (g *GPU) loadF32(u *uploader, f *gguf.File, r io.ReaderAt, name string) (*webgpu.Buffer, error) {
	t, ok := f.Tensor(name)
	if !ok || t.Type != gguf.TypeF32 {
		return nil, fmt.Errorf("llm: tensor %s missing or not F32", name)
	}
	b, err := g.buffer(t.Size, stor)
	if err != nil {
		return nil, err
	}
	data := make([]byte, t.Size)
	if _, err := r.ReadAt(data, int64(f.DataOffset+t.Offset)); err != nil {
		return nil, err
	}
	return b, u.write(b, 0, data)
}

func (g *GPU) grid(rows int) (x, y uint32) {
	mx := int(g.lim.MaxWorkgroupsPerDimension)
	x = uint32(min(rows, mx))
	y = uint32((rows + int(x) - 1) / int(x))
	return
}

// mvBind makes the bind group of a matrix product y (+)= W x.
func (g *GPU) mvBind(u *uploader, m *mat, x, y *webgpu.Buffer, acc bool) (*webgpu.BindGroup, error) {
	nx, _ := g.grid(m.out)
	a := uint32(0)
	if acc {
		a = 1
	}
	meta, err := g.metaBuf(u, uint32(m.in), uint32(m.out), a, nx)
	if err != nil {
		return nil, err
	}
	var (
		bg   *webgpu.BindGroup
		kn   = "q80"
		err2 error
	)
	switch m.typ {
	case gguf.TypeQ4_0:
		kn = "q40"
		bg, err2 = g.bind(kn, m.qs, m.d, x, y, meta)
	case gguf.TypeQ4_1:
		kn = "q41"
		bg, err2 = g.bind(kn, m.qs, m.d, m.m, x, y, meta)
	default:
		bg, err2 = g.bind(kn, m.qs, m.d, x, y, meta)
	}
	if err2 == nil {
		g.kinds[bg] = kn
	}
	return bg, err2
}

// NewGPU creates the kernels, allocates every buffer and streams the weights
// of the model to the GPU. ctx is the KV cache length in tokens.
func NewGPU(dev *webgpu.Device, lim Limits, f *gguf.File, r io.ReaderAt, ctx int) (*GPU, error) {
	cfg, err := ConfigFrom(f)
	if err != nil {
		return nil, err
	}
	if ctx <= 0 || ctx > cfg.Ctx {
		ctx = cfg.Ctx
	}
	g := &GPU{dev: dev, lim: lim, Cfg: cfg, ctx: ctx, wg: lim.WorkgroupSize(), named: map[string]*webgpu.Buffer{}, fgrid: map[string]fusedGrid{}, k: map[string]*kernel{}, kinds: map[*webgpu.BindGroup]string{}}
	if lim.MaxWorkgroupsPerDimension == 0 || lim.MaxStorageBuffers < 8 {
		return nil, errors.New("llm: adapter limits below WebGPU's defaults")
	}
	for _, k := range []struct{ n, src, acc string }{
		{"embed", kernelEmbed, "rrrwrr"}, {"rms", kernelRMS, "rrwr"},
		{"q40", kernelMatvecQ40, "rrrwr"}, {"q41", kernelMatvecQ41, "rrrrwr"}, {"q80", kernelMatvecQ80, "rrrwr"},
		{"rope", kernelRopeKV, "wrrwwrrr"}, {"attn", kernelAttn, "rrrwwrr"}, {"swi", kernelSwiGLU, "wrr"},
		{"argmax", kernelArgmax, "rwwrr"}, {"adv", kernelAdvance, "w"},
	} {
		if err := g.newKernel(k.n, k.src, k.acc); err != nil {
			return nil, err
		}
	}
	c := cfg
	kvd := c.KVHeads * c.HeadDim
	u := &uploader{g: g}
	mk := func(n int) (*webgpu.Buffer, error) { return g.buffer(uint64(4*n), stor) }
	var bufs [14]*webgpu.Buffer
	// A prompt chunk of pm tokens goes through the fused layers together (one
	// dispatch per kernel for the chunk): its residual stream, queries and
	// activations are pm vectors wide.
	g.pm = 1
	if !NoFused && g.fusedOK() {
		all := true
		for l := 0; l < c.Layers; l++ {
			all = all && layerFusable(f, fmt.Sprintf("blk.%d.", l))
		}
		if all {
			// The chunk's attention partials are pm * heads * chunks * HeadDim
			// floats: at most 16 MiB.
			per := 4 * c.Heads * g.chunks() * c.HeadDim
			g.pm = max(1, min(PrefillBatch, int(lim.MaxWorkgroupsPerDimension), (16<<20)/per))
		}
	}
	for i, n := range []int{c.Embd * g.pm, c.Embd, c.Embd * g.pm, kvd, kvd, c.Embd, c.FF * g.pm, c.FF, c.Vocab, 2, ctx, 1, c.Heads * ctx} {
		if bufs[i], err = mk(n); err != nil {
			return nil, err
		}
	}
	x, h, q, kk, vv, att, gt, up, logits, state, tokens, outb, scratch := bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7], bufs[8], bufs[9], bufs[10], bufs[11], bufs[12]
	g.state, g.tokens, g.outb, g.logits, g.xbuf = state, tokens, outb, logits, x
	for n, b := range map[string]*webgpu.Buffer{"x": x, "h": h, "q": q, "k": kk, "v": vv, "att": att, "gate": gt, "up": up, "logits": logits, "state": state, "tokens": tokens, "outb": outb} {
		g.named[n] = b
	}
	if g.outRead, err = g.buffer(4*MaxBatch, webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if g.logitsRead, err = g.buffer(uint64(4*c.Vocab), webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	// RoPE table: cos and sin of pos * base^(-2i/dim) in double precision.
	hp := c.HeadDim / 2
	cs := make([]byte, 0, ctx*hp*8)
	for pos := 0; pos < ctx; pos++ {
		for i := 0; i < hp; i++ {
			theta := float64(pos) * math.Pow(float64(c.RopeBase), -float64(2*i)/float64(c.HeadDim))
			s, co := math.Sincos(theta)
			cs = binary.LittleEndian.AppendUint32(cs, math.Float32bits(float32(co)))
			cs = binary.LittleEndian.AppendUint32(cs, math.Float32bits(float32(s)))
		}
	}
	csb, err := g.buffer(uint64(len(cs)), stor)
	if err != nil {
		return nil, err
	}
	if err := u.write(csb, 0, cs); err != nil {
		return nil, err
	}
	// embedding (Q8_0) and its bind group
	emb, err := g.loadMat(u, f, r, "token_embd.weight")
	if err != nil {
		return nil, err
	}
	if emb.typ != gguf.TypeQ8_0 {
		return nil, errors.New("llm: token_embd.weight must be Q8_0")
	}
	embMeta, err := g.metaBuf(u, uint32(c.Embd))
	if err != nil {
		return nil, err
	}
	if g.embed, err = g.bind("embed", tokens, emb.qs, emb.d, x, state, embMeta); err != nil {
		return nil, err
	}
	rmsMeta, err := g.metaBuf(u, uint32(c.Embd), math.Float32bits(c.Eps))
	if err != nil {
		return nil, err
	}
	nw, err := g.loadF32(u, f, r, "output_norm.weight")
	if err != nil {
		return nil, err
	}
	if g.pm > 1 {
		// The last token of a prompt chunk is copied to xlast (a buffer cannot
		// be copied onto itself) and the head reads it.
		if g.xlast, err = mk(c.Embd); err != nil {
			return nil, err
		}
		if g.rmsL, err = g.bind("rms", g.xlast, nw, h, rmsMeta); err != nil {
			return nil, err
		}
	}
	if g.rmsF, err = g.bind("rms", x, nw, h, rmsMeta); err != nil {
		return nil, err
	}
	if g.logitB, err = g.mvBind(u, emb, h, logits, false); err != nil {
		return nil, err
	}
	argMeta, err := g.metaBuf(u, uint32(c.Vocab))
	if err != nil {
		return nil, err
	}
	if g.argmax, err = g.bind("argmax", logits, tokens, outb, state, argMeta); err != nil {
		return nil, err
	}
	if g.adv, err = g.bind("adv", state); err != nil {
		return nil, err
	}
	if g.pm > 1 {
		if err := g.newKernel("advn", kernelAdvanceN, "wr"); err != nil {
			return nil, err
		}
		if g.advNA, err = g.buffer(4, stor); err != nil {
			return nil, err
		}
		one, err := g.metaBuf(u, 1)
		if err != nil {
			return nil, err
		}
		if g.advA, err = g.bind("advn", state, g.advNA); err != nil {
			return nil, err
		}
		if g.advB, err = g.bind("advn", state, one); err != nil {
			return nil, err
		}
	}
	if !NoFused {
		a1, a2 := argmaxSources(g.wg, c.Vocab)
		if err := g.newKernel("am1", a1, "rww"); err != nil {
			return nil, err
		}
		if err := g.newKernel("am2", a2, "rrwwr"); err != nil {
			return nil, err
		}
		pbv, err := g.buffer(4*argmaxGroups, stor)
		if err != nil {
			return nil, err
		}
		pbi, err := g.buffer(4*argmaxGroups, stor)
		if err != nil {
			return nil, err
		}
		if g.am1, err = g.bind("am1", logits, pbv, pbi); err != nil {
			return nil, err
		}
		if g.am2, err = g.bind("am2", pbv, pbi, tokens, outb, state); err != nil {
			return nil, err
		}
	}
	ropeMeta, err := g.metaBuf(u, uint32(c.Heads), uint32(c.KVHeads), uint32(c.HeadDim), uint32(ctx))
	if err != nil {
		return nil, err
	}
	attnMeta, err := g.metaBuf(u, uint32(c.Heads), uint32(c.KVHeads), uint32(c.HeadDim), uint32(ctx), math.Float32bits(float32(1/math.Sqrt(float64(c.HeadDim)))))
	if err != nil {
		return nil, err
	}
	swiMeta, err := g.metaBuf(u, uint32(c.FF))
	if err != nil {
		return nil, err
	}
	for l := 0; l < c.Layers; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		var lb layerBG
		if !NoFused && g.fusedOK() && layerFusable(f, p) {
			dt, _ := f.Tensor(p + "ffn_down.weight")
			if g.pv == nil {
				if g.pv, err = g.buffer(uint64(4*c.Heads*g.chunks()*c.HeadDim*g.pm), stor); err != nil {
					return nil, err
				}
				if g.ml, err = g.buffer(uint64(8*c.Heads*g.chunks()*g.pm), stor); err != nil {
					return nil, err
				}
			}
			if err := g.fusedKernels(dt.Type == gguf.TypeQ4_1); err != nil {
				return nil, err
			}
			if lb.fx, err = g.fusedLayer(u, f, r, p, x, q, gt, g.pv, g.ml, state, csb); err != nil {
				return nil, err
			}
			g.layers = append(g.layers, lb)
			continue
		}
		an, err := g.loadF32(u, f, r, p+"attn_norm.weight")
		if err != nil {
			return nil, err
		}
		fn, err := g.loadF32(u, f, r, p+"ffn_norm.weight")
		if err != nil {
			return nil, err
		}
		kc, err := g.buffer(uint64(4*ctx*kvd), stor)
		if err != nil {
			return nil, err
		}
		vc, err := g.buffer(uint64(4*ctx*kvd), stor)
		if err != nil {
			return nil, err
		}
		type use struct {
			name string
			x, y *webgpu.Buffer
			acc  bool
			dst  **webgpu.BindGroup
		}
		for _, w := range []use{
			{"attn_q.weight", h, q, false, &lb.q}, {"attn_k.weight", h, kk, false, &lb.k}, {"attn_v.weight", h, vv, false, &lb.v},
			{"attn_output.weight", att, x, true, &lb.o}, {"ffn_gate.weight", h, gt, false, &lb.gate},
			{"ffn_up.weight", h, up, false, &lb.up}, {"ffn_down.weight", gt, x, true, &lb.down},
		} {
			m, err := g.loadMat(u, f, r, p+w.name)
			if err != nil {
				return nil, err
			}
			if *w.dst, err = g.mvBind(u, m, w.x, w.y, w.acc); err != nil {
				return nil, err
			}
		}
		if lb.rms1, err = g.bind("rms", x, an, h, rmsMeta); err != nil {
			return nil, err
		}
		if lb.rms2, err = g.bind("rms", x, fn, h, rmsMeta); err != nil {
			return nil, err
		}
		if lb.rope, err = g.bind("rope", q, kk, vv, kc, vc, state, csb, ropeMeta); err != nil {
			return nil, err
		}
		if lb.attn, err = g.bind("attn", q, kc, vc, scratch, att, state, attnMeta); err != nil {
			return nil, err
		}
		if lb.swi, err = g.bind("swi", gt, up, swiMeta); err != nil {
			return nil, err
		}
		g.layers = append(g.layers, lb)
	}
	if err := u.flush(); err != nil {
		return nil, err
	}
	return g, nil
}

// MemoryUsed is the GPU memory the model, its cache and its scratch hold.
func (g *GPU) MemoryUsed() uint64 { return g.used }

// Skip names the kernel classes the pass leaves out ("all" for every one): a
// profiling aid. GPU time per class is the decode time with the class minus
// the decode time without it; the output is garbage then.
func (g *GPU) dispatch(p *webgpu.ComputePassRecorder, kn string, bg *webgpu.BindGroup, x, y uint32) {
	g.dispatch3(p, kn, bg, x, y, 1)
}

func (g *GPU) dispatch3(p *webgpu.ComputePassRecorder, kn string, bg *webgpu.BindGroup, x, y, z uint32) {
	if g.Skip != nil && (g.Skip[kn] || g.Skip["all"]) {
		return
	}
	p.SetPipeline(g.k[kn].pipe)
	p.SetBindGroup(0, bg)
	p.DispatchWorkgroups(x, y, z)
	g.Stats.Dispatches++
}

func (g *GPU) groups(threads int) (x, y uint32) {
	n := (threads + g.wg - 1) / g.wg
	return g.grid(n)
}

// token records the forward pass of one token (the one at tokens[step]).
func (g *GPU) token(p *webgpu.ComputePassRecorder, head bool) {
	c := g.Cfg
	kvd := c.KVHeads * c.HeadDim
	g.dispatch(p, "embed", g.embed, 1, 1)
	mvx := func(rows int) (uint32, uint32) { return g.grid(rows) }
	rx, ry := g.groups((c.Heads+c.KVHeads)*c.HeadDim/2 + kvd)
	sx, sy := g.groups(c.FF)
	layers := g.layers
	if g.Layers != 0 && g.Layers < len(layers) {
		layers = layers[:max(g.Layers, 0)]
		head = false
	}
	for _, l := range layers {
		if l.fx != nil {
			g.recordFused(p, l.fx, 1)
			continue
		}
		g.dispatch(p, "rms", l.rms1, 1, 1)
		x, y := mvx(c.Embd)
		g.dispatch(p, g.mvk(l.q), l.q, x, y)
		x, y = mvx(kvd)
		g.dispatch(p, g.mvk(l.k), l.k, x, y)
		g.dispatch(p, g.mvk(l.v), l.v, x, y)
		g.dispatch(p, "rope", l.rope, rx, ry)
		g.dispatch(p, "attn", l.attn, uint32(c.Heads), 1)
		x, y = mvx(c.Embd)
		g.dispatch(p, g.mvk(l.o), l.o, x, y)
		g.dispatch(p, "rms", l.rms2, 1, 1)
		x, y = mvx(c.FF)
		g.dispatch(p, g.mvk(l.gate), l.gate, x, y)
		g.dispatch(p, g.mvk(l.up), l.up, x, y)
		g.dispatch(p, "swi", l.swi, sx, sy)
		x, y = mvx(c.Embd)
		g.dispatch(p, g.mvk(l.down), l.down, x, y)
	}
	if head {
		g.recordHead(p, false)
	}
	g.dispatch(p, "adv", g.adv, 1, 1)
	g.pos++
}

// mvk returns the kernel name of a layer's bind group; the matrix types of
// a layer are known when the bind groups are made, so remember them.
func (g *GPU) mvk(bg *webgpu.BindGroup) string { return g.kinds[bg] }

// reset zeroes the position and step and loads the prompt into tokens[0:].
func (g *GPU) begin(rec *webgpu.Recorder, prompt []int32) {
	rec.WriteBuffer(g.state, 0, le32(0, 0))
	g.pos = 0
	ids := make([]uint32, len(prompt))
	for i, t := range prompt {
		ids[i] = uint32(t)
	}
	rec.WriteBuffer(g.tokens, 0, le32(ids...))
}

// PrefillChunk is the number of prompt tokens recorded per submit: a submit
// must stay far below the host's 2 s limit.
const PrefillChunk = 8

func (g *GPU) prefill(prompt []int32, lastRead func(rec *webgpu.Recorder) func() error) (func() error, error) {
	if len(prompt) == 0 {
		return nil, errors.New("llm: empty prompt")
	}
	var fin func() error
	if g.pm > 1 {
		for a := 0; a < len(prompt); a += g.pm {
			b := min(a+g.pm, len(prompt))
			rec := g.dev.NewRecorder()
			if a == 0 {
				g.begin(rec, prompt)
			}
			if err := g.chunk(rec, b-a, b == len(prompt)); err != nil {
				return nil, err
			}
			if b == len(prompt) {
				fin = lastRead(rec)
			}
			if _, err := rec.Submit(); err != nil {
				return nil, err
			}
			g.Stats.Submits++
		}
		return fin, nil
	}
	for a := 0; a < len(prompt); a += PrefillChunk {
		b := min(a+PrefillChunk, len(prompt))
		rec := g.dev.NewRecorder()
		if a == 0 {
			g.begin(rec, prompt)
		}
		p := rec.BeginComputePass()
		for i := a; i < b; i++ {
			g.token(p, i == len(prompt)-1)
		}
		if err := p.End(); err != nil {
			return nil, err
		}
		if b == len(prompt) {
			fin = lastRead(rec)
		}
		if _, err := rec.Submit(); err != nil {
			return nil, err
		}
		g.Stats.Submits++
	}
	return fin, nil
}

// PrefillLogits runs the prompt and returns the logits of its last token
// (for tests: reading 4*vocab bytes per token is what Generate avoids).
func (g *GPU) PrefillLogits(prompt []int32) ([]float32, error) {
	if len(prompt) >= g.ctx {
		return nil, errors.New("llm: prompt longer than the context")
	}
	var rb *webgpu.Readback
	fin, err := g.prefill(prompt, func(rec *webgpu.Recorder) func() error {
		rec.CopyBufferToBuffer(g.logits, 0, g.logitsRead, 0, uint64(4*g.Cfg.Vocab))
		rb = rec.ReadBuffer(g.logitsRead, 0, uint64(4*g.Cfg.Vocab))
		return func() error { return nil }
	})
	if err != nil {
		return nil, err
	}
	_ = fin
	b, err := rb.Bytes()
	if err != nil {
		return nil, err
	}
	out := make([]float32, g.Cfg.Vocab)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}

// Timing of a generation.
type Timing struct {
	Prefill, Decode time.Duration
	PromptTokens    int
	Generated       int
	Submits         int
	Dispatches      int
	// Decode time split: the guest recording the pass (Go, in wasm), and
	// Submit (the crossing, the replay, the GPU, the read-back).
	Record, Submit time.Duration
}

// Generate decodes greedily: the prompt in submits of PrefillChunk tokens,
// then one submit per token whose only read-back is the 4-byte sampled token
// (argmax on the GPU). emit is called with each generated token; Generate
// stops after n tokens, at the end-of-sequence token (which is not emitted)
// or when emit returns false.
func (g *GPU) Generate(prompt []int32, n int, eos int32, emit func(int32) bool) (Timing, error) {
	var tm Timing
	if len(prompt)+n > g.ctx {
		return tm, fmt.Errorf("llm: prompt of %d tokens and %d new tokens exceed the context of %d", len(prompt), n, g.ctx)
	}
	s0, d0 := g.Stats.Submits, g.Stats.Dispatches
	t0 := time.Now()
	var rb *webgpu.Readback
	_, err := g.prefill(prompt, func(rec *webgpu.Recorder) func() error {
		rec.CopyBufferToBuffer(g.outb, 0, g.outRead, 0, 4)
		rb = rec.ReadBuffer(g.outRead, 0, 4)
		return nil
	})
	if err != nil {
		return tm, err
	}
	tm.Prefill = time.Since(t0)
	tm.PromptTokens = len(prompt)
	t1 := time.Now()
	cnt := 1 // tokens the pending read-back holds
	for produced := 0; produced < n; {
		b, err := rb.Bytes()
		if err != nil {
			return tm, err
		}
		for j := 0; j < cnt && produced < n; j++ {
			tok := int32(binary.LittleEndian.Uint32(b[4*j:]))
			if tok == eos {
				n = produced // stop; the tokens the GPU went on to compute are dropped
				break
			}
			tm.Generated++
			produced++
			if !emit(tok) {
				n = produced
				break
			}
		}
		if produced >= n {
			break
		}
		// Next submit: up to Batch tokens, each sampled on the GPU from the
		// one before, so the 0.4 ms crossing and the read-back are paid once
		// for Batch tokens. The sampled tokens are contiguous in tokens[].
		cnt = min(g.batch(), n-produced)
		ta := time.Now()
		rec := g.dev.NewRecorder()
		p := rec.BeginComputePass()
		first := g.pos
		for j := 0; j < cnt; j++ {
			g.token(p, true)
		}
		if err := p.End(); err != nil {
			return tm, err
		}
		rec.CopyBufferToBuffer(g.tokens, uint64(4*(first+1)), g.outRead, 0, uint64(4*cnt))
		rb = rec.ReadBuffer(g.outRead, 0, uint64(4*cnt))
		tb := time.Now()
		if _, err := rec.Submit(); err != nil {
			return tm, err
		}
		tm.Record += tb.Sub(ta)
		tm.Submit += time.Since(tb)
		g.Stats.Submits++
	}
	tm.Decode = time.Since(t1)
	tm.Submits, tm.Dispatches = g.Stats.Submits-s0, g.Stats.Dispatches-d0
	return tm, nil
}

// Peek reads the first n floats of an activation buffer ("x", "h", "q", "k",
// "v", "att", "gate", "up", "logits"): for tests, never in the decode loop.
func (g *GPU) Peek(name string, n int) ([]float32, error) {
	b, ok := g.named[name]
	if !ok {
		return nil, fmt.Errorf("llm: no buffer %q", name)
	}
	tmp, err := g.dev.CreateBuffer(webgpu.BufferDescriptor{Size: uint64(4 * n), Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	rec := g.dev.NewRecorder()
	rec.CopyBufferToBuffer(b, 0, tmp, 0, uint64(4*n))
	rb := rec.ReadBuffer(tmp, 0, uint64(4*n))
	if _, err := rec.Submit(); err != nil {
		return nil, err
	}
	g.Stats.Submits++
	raw, err := rb.Bytes()
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out, nil
}

// MaxBatch bounds Batch.
const MaxBatch = 16

// DecodeBatch is the number of tokens a decode submit computes (Generate): the
// crossing and the read-back are paid once for that many tokens. After the
// end-of-sequence token or a stop from emit, at most DecodeBatch-1 tokens
// have been computed for nothing.
var DecodeBatch = 8

func (g *GPU) batch() int { return max(1, min(DecodeBatch, MaxBatch)) }
