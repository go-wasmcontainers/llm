package llm

import (
	"strconv"
	"strings"
)

// Fused decode kernels (v0.3.0). One layer of the pass is five dispatches:
//
//	K1  RMSNorm + Q,K,V matrix products + RoPE + KV-cache append
//	K2  attention (all heads)
//	K3  attention output projection, added to the residual
//	K4  RMSNorm + gate and up products + SiLU(gate)*up
//	K5  down projection, added to the residual
//
// Every kernel has its model dimensions baked in as WGSL constants (the
// shaders are generated per model), so loop bounds are constants and no
// parameter buffer is bound. A workgroup computes RPW = WG/TPR output rows,
// TPR invocations per row, each taking blocks b = lane, lane+TPR, ... of the
// row's quantised blocks (one block = one 16-byte vec4<u32> load); the input
// vector is staged once per workgroup in workgroup memory (in tiles when it
// does not fit the adapter's workgroup-memory limit); the TPR partial sums
// of a row are added by a tree reduction in workgroup memory. No subgroups,
// no f16: the summation order depends on WG and TPR only.

const fusedMV = `
const WG: u32 = {{WG}}u;
const TPR: u32 = {{TPR}}u;
const RPW: u32 = {{RPW}}u;
const N: u32 = {{N}}u;
const NB: u32 = {{NB}}u;
const ROWS: u32 = {{ROWS}}u;
const NX: u32 = {{NX}}u;
const TILE4: u32 = {{TILE4}}u;
const NTILES: u32 = {{NTILES}}u;
const TB: u32 = {{TB}}u;
const EPS: f32 = {{EPS}};
{{DECL}}
var<workgroup> xs: array<vec4<f32>, {{TILE4}}>;

{{BLOCKFN}}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let wg = wid.y * NX + wid.x;
  let lr = li / TPR;
  let ln = li % TPR;
  {{ROWMAP}}
  let r = select(0u, row, valid);
  {{PRE}}
  var acc = 0.0;
  for (var t = 0u; t < NTILES; t = t + 1u) {
    let base4 = t * TILE4;
    var i = li;
    loop {
      if (i >= TILE4) { break; }
      let gi = base4 + i;
      var v = vec4<f32>(0.0);
      if (gi < N / 4u) { v = {{LOAD}}; }
      xs[i] = v;
      i = i + WG;
    }
    workgroupBarrier();
    var b = ln;
    loop {
      if (b >= TB) { break; }
      let gb = t * TB + b;
      if (gb < NB) {
        let xb = b * 8u;
        {{BLOCK}}
      }
      b = b + TPR;
    }
    workgroupBarrier();
  }
  red[li] = acc;
  workgroupBarrier();
  var s = TPR / 2u;
  loop {
    if (s == 0u) { break; }
    if (ln < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  {{EPI}}
}
`

// Q4_0 block: byte i holds element i (low nibble) and element i+16 (high
// nibble) of the 32 of the block, as in ggml.
const fusedFnQ40 = `
fn nib(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)); }
fn nibh(w: u32) -> vec4<f32> { return nib(w >> 4u); }
`

const fusedBlockQ40 = `
        let q = qs[r * NB + gb];
        let sc = {{SCALE}}[r * NB + gb];
        var s4 = vec4<f32>(0.0);
        s4 = s4 + (nib(q.x) - 8.0) * xs[xb] + (nibh(q.x) - 8.0) * xs[xb + 4u];
        s4 = s4 + (nib(q.y) - 8.0) * xs[xb + 1u] + (nibh(q.y) - 8.0) * xs[xb + 5u];
        s4 = s4 + (nib(q.z) - 8.0) * xs[xb + 2u] + (nibh(q.z) - 8.0) * xs[xb + 6u];
        s4 = s4 + (nib(q.w) - 8.0) * xs[xb + 3u] + (nibh(q.w) - 8.0) * xs[xb + 7u];
        acc = acc + sc * (s4.x + s4.y + s4.z + s4.w);
`

// Q4_1: value = d*q + m, so a block's sum is d*sum(q*x) + m*sum(x).
const fusedBlockQ41 = `
        let q = qs[r * NB + gb];
        var s4 = vec4<f32>(0.0);
        s4 = s4 + nib(q.x) * xs[xb] + nibh(q.x) * xs[xb + 4u];
        s4 = s4 + nib(q.y) * xs[xb + 1u] + nibh(q.y) * xs[xb + 5u];
        s4 = s4 + nib(q.z) * xs[xb + 2u] + nibh(q.z) * xs[xb + 6u];
        s4 = s4 + nib(q.w) * xs[xb + 3u] + nibh(q.w) * xs[xb + 7u];
        let sx = xs[xb] + xs[xb + 1u] + xs[xb + 2u] + xs[xb + 3u] + xs[xb + 4u] + xs[xb + 5u] + xs[xb + 6u] + xs[xb + 7u];
        acc = acc + dd[r * NB + gb] * (s4.x + s4.y + s4.z + s4.w) + mm[r * NB + gb] * (sx.x + sx.y + sx.z + sx.w);
`

const fusedPreRMS = `
  var ss = 0.0;
  var si = li;
  loop {
    if (si >= N / 4u) { break; }
    let v = x4[si];
    ss = ss + dot(v, v);
    si = si + WG;
  }
  let rs = 1.0 / sqrt(rsum(li, ss) / f32(N) + EPS);
`

const fusedLoadRMS = `x4[gi] * rs * vec4<f32>(aux[NW + 4u * gi], aux[NW + 4u * gi + 1u], aux[NW + 4u * gi + 2u], aux[NW + 4u * gi + 3u])`

// Row maps. Plain: row = wg*RPW + lr. Gate/up: a workgroup computes RPW/2
// rows of gate and the same rows of up (the matrices are concatenated
// [gate; up], F rows each).
const fusedRowPlain = `let row = wg * RPW + lr; let valid = row < ROWS;`
const fusedRowGateUp = `
  let hh = RPW / 2u;
  let orow = wg * hh + lr % hh;
  let valid = orow < F;
  let row = select(orow, F + orow, lr >= hh);`

const fusedEpiStore = `
  if (ln == 0u && valid) { y[row] = red[li]; }`
const fusedEpiAcc = `
  if (ln == 0u && valid) { y[row] = y[row] + red[li]; }`

const fusedEpiGateUp = `
  if (li < RPW / 2u) {
    let o = wg * (RPW / 2u) + li;
    if (o < F) {
      let g = red[li * TPR];
      let u = red[(li + RPW / 2u) * TPR];
      act[o] = g / (1.0 + exp(-g)) * u;
    }
  }`

// RoPE on adjacent pairs (rows 2p, 2p+1 of a head, both in this workgroup
// since RPW is even) from the double-precision table, then K and V go to the
// cache (K at [pos*KVD], V at [CTXKVD + pos*KVD]).
const fusedEpiQKV = `
  if (li < RPW / 2u) {
    let rr = wg * RPW + 2u * li;
    if (rr < ROWS) {
      let a = red[(2u * li) * TPR];
      let b = red[(2u * li + 1u) * TPR];
      let pos = state[0];
      if (rr < NQ + NK) {
        var j = rr;
        if (rr >= NQ) { j = rr - NQ; }
        let p = (j % HD) / 2u;
        let c = cs[(pos * HP + p) * 2u];
        let sn = cs[(pos * HP + p) * 2u + 1u];
        let ra = a * c - b * sn;
        let rb = a * sn + b * c;
        if (rr < NQ) {
          q[rr] = ra; q[rr + 1u] = rb;
        } else {
          kv[pos * KVD + j] = ra; kv[pos * KVD + j + 1u] = rb;
        }
      } else {
        let j = rr - NQ - NK;
        kv[CTXKVD + pos * KVD + j] = a;
        kv[CTXKVD + pos * KVD + j + 1u] = b;
      }
    }
  }`

// Attention, all heads of one query position. A workgroup per head: scores
// (one position per invocation, vec4 loads), softmax with a max subtraction,
// then the weighted sum of V with the context split over G = WG/HD4 groups of
// HD4 invocations (one vec4 of the head per invocation) and a final add of
// the G partials.
const fusedAttn = wgslPrelude + `
const HD4: u32 = {{HD4}}u;
const KVD4: u32 = {{KVD4}}u;
const CTXKVD4: u32 = {{CTXKVD4}}u;
const CTX: u32 = {{CTX}}u;
const GQ: u32 = {{GQ}}u;
const G: u32 = {{G}}u;
const WG: u32 = {{WG}}u;
const S: u32 = {{S}}u;
const SCALE: f32 = {{SCALE}};
@group(0) @binding(0) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pv: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ml: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
var<workgroup> qh: array<vec4<f32>, {{HD4}}>;
var<workgroup> pw: array<f32, {{WG}}>;
var<workgroup> part: array<vec4<f32>, {{WG}}>;

// One workgroup per (chunk of WG positions, head): the partial softmax of the
// chunk (its max m and sum l of exp(s-m)) and its weighted sum of V go to
// ml and pv; K3 merges the chunks of a head (flash-decoding: Dao et al.), so
// the work of a head is spread over ceil(n/WG) workgroups and no loop of an
// invocation is longer than HD4.
@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = wid.x;
  let h = wid.y;
  let kvh = h / GQ;
  let n = state[0] + 1u;
  let p = c * WG + li;
  if (li < HD4) { qh[li] = q4[h * HD4 + li]; }
  workgroupBarrier();
  var s = -3.0e38;
  if (p < n) {
    var a = 0.0;
    let kb = p * KVD4 + kvh * HD4;
    for (var d = 0u; d < HD4; d = d + 1u) {
      a = a + dot(qh[d], kv4[kb + d]);
    }
    s = a * SCALE;
  }
  let m = rmax(li, s);
  let e = select(0.0, exp(s - m), p < n);
  pw[li] = e;
  let l = rsum(li, e);
  let lane = li % HD4;
  var o = vec4<f32>(0.0);
  var u = li / HD4;
  loop {
    if (u >= WG) { break; }
    let pp = c * WG + u;
    if (pp < n) {
      o = o + pw[u] * kv4[CTXKVD4 + pp * KVD4 + kvh * HD4 + lane];
    }
    u = u + G;
  }
  part[li] = o;
  workgroupBarrier();
  if (li < HD4) {
    var tot = part[li];
    for (var k = 1u; k < G; k = k + 1u) { tot = tot + part[li + k * HD4]; }
    pv[(h * S + c) * HD4 + li] = tot;
  }
  if (li == 0u) { ml[h * S + c] = vec2<f32>(m, l); }
}
`

// The merge of the chunks of a head, as the input stage of K3: element gi (a
// vec4 of the concatenated heads) is sum_c exp(m_c-M) pv_c / sum_c exp(m_c-M) l_c.
const fusedCombine = `
fn combine(gi: u32) -> vec4<f32> {
  let h = gi / HD4;
  let d4 = gi % HD4;
  let sn = (state[0] + WG) / WG;
  var mx = -3.0e38;
  for (var c = 0u; c < S; c = c + 1u) {
    if (c >= sn) { break; }
    mx = max(mx, ml[h * S + c].x);
  }
  var l = 0.0;
  var o = vec4<f32>(0.0);
  for (var c = 0u; c < S; c = c + 1u) {
    if (c >= sn) { break; }
    let w = exp(ml[h * S + c].x - mx);
    l = l + w * ml[h * S + c].y;
    o = o + w * pv[(h * S + c) * HD4 + d4];
  }
  return o / l;
}
`

type mvSpec struct {
	wg, tpr   int
	maxShared int
	n, rows   int
	maxGroups int
	eps       float32
}

func f32lit(x float32) string { return strconv.FormatFloat(float64(x), 'e', -1, 32) }

// mvKernel builds one fused matrix-vector kernel. kind is "q40" or "q41";
// mode is "qkv", "gateup", "acc" or "store"; norm adds the RMSNorm (weights
// at aux[nw:]) in front.
type mvKind struct {
	q41    bool
	mode   string
	norm   bool
	attnIn bool // the input is the merge of the attention chunks (fusedCombine)
}

func (s mvSpec) tile() (tile4, ntiles, tb int) {
	// Workgroup memory: the reduction array (WG floats) and the staged tile.
	avail := s.maxShared - s.wg*4 - s.wg*16 - 512
	t := s.n
	if t*4 > avail {
		t = avail / 4 / 32 * 32
	}
	if t < 32 {
		t = 32
	}
	return t / 4, (s.n + t - 1) / t, t / 32
}

// fusedMVSource returns the WGSL of a fused matrix-vector kernel, with extra
// constants (name, value) for the epilogue, and its binding access string.
func fusedMVSource(s mvSpec, k mvKind, extra map[string]int) (src, access string, nwgs int) {
	rpw := s.wg / s.tpr
	rows := s.rows
	if k.mode == "gateup" {
		nwgs = (extra["F"] + rpw/2 - 1) / (rpw / 2)
	} else {
		nwgs = (rows + rpw - 1) / rpw
	}
	nx := min(nwgs, s.maxGroups)
	t4, nt, tb := s.tile()
	nb := s.n / 32
	var decl, blockFn, pre, load, rowmap, epi, block string
	// bindings
	b := 0
	bind := func(name, typ string, rw bool) string {
		a := "read"
		if rw {
			a = "read_write"
		}
		v := "@group(0) @binding(" + itoa(b) + ") var<storage, " + a + "> " + name + ": " + typ + ";\n"
		b++
		if rw {
			access += "w"
		} else {
			access += "r"
		}
		return v
	}
	decl += bind("qs", "array<vec4<u32>>", false)
	scale := "dd"
	if k.norm {
		decl += bind("aux", "array<f32>", false)
		scale = "aux"
		decl += "const NW: u32 = " + itoa(extra["NW"]) + "u;\n"
	} else {
		decl += bind("dd", "array<f32>", false)
	}
	if k.q41 {
		decl += bind("mm", "array<f32>", false)
		blockFn = fusedFnQ40
		block = fusedBlockQ41
	} else {
		blockFn = fusedFnQ40
		block = fusedBlockQ40
	}
	if k.norm {
		decl += bind("x4", "array<vec4<f32>>", false)
		pre = fusedPreRMS
		load = fusedLoadRMS
	} else if k.attnIn {
		decl += bind("pv", "array<vec4<f32>>", false)
		decl += bind("ml", "array<vec2<f32>>", false)
		decl += bind("state", "array<u32>", false)
		decl += "const HD4: u32 = " + itoa(extra["HD4"]) + "u;\nconst S: u32 = " + itoa(extra["S"]) + "u;\n"
		blockFn += fusedCombine
		load = "combine(gi)"
	} else {
		decl += bind("x4", "array<vec4<f32>>", false)
		load = "x4[gi]"
	}
	switch k.mode {
	case "qkv":
		decl += bind("q", "array<f32>", true)
		decl += bind("kv", "array<f32>", true)
		decl += bind("state", "array<u32>", false)
		decl += bind("cs", "array<f32>", false)
		for _, n := range []string{"NQ", "NK", "HD", "HP", "KVD", "CTXKVD"} {
			decl += "const " + n + ": u32 = " + itoa(extra[n]) + "u;\n"
		}
		rowmap, epi = fusedRowPlain, fusedEpiQKV
	case "gateup":
		decl += bind("act", "array<f32>", true)
		decl += "const F: u32 = " + itoa(extra["F"]) + "u;\n"
		rowmap, epi = fusedRowGateUp, fusedEpiGateUp
	case "acc":
		decl += bind("y", "array<f32>", true)
		rowmap, epi = fusedRowPlain, fusedEpiAcc
	default:
		decl += bind("y", "array<f32>", true)
		rowmap, epi = fusedRowPlain, fusedEpiStore
	}
	src = wgslPrelude + fusedMV
	block = strings.ReplaceAll(block, "{{SCALE}}", scale)
	r := strings.NewReplacer(
		"{{DECL}}", decl, "{{BLOCKFN}}", blockFn, "{{PRE}}", pre, "{{LOAD}}", load, "{{ROWMAP}}", rowmap, "{{EPI}}", epi, "{{BLOCK}}", block,
		"{{RPW}}", itoa(rpw), "{{TPR}}", itoa(s.tpr), "{{N}}", itoa(s.n), "{{NB}}", itoa(nb), "{{ROWS}}", itoa(rows),
		"{{NX}}", itoa(nx), "{{TILE4}}", itoa(t4), "{{NTILES}}", itoa(nt), "{{TB}}", itoa(tb), "{{EPS}}", f32lit(s.eps), "{{WG}}", itoa(s.wg))
	return r.Replace(src), access, nwgs
}

func fusedAttnSource(wg, hd, heads, kvheads, ctx int, scale float32) string {
	kvd := kvheads * hd
	return strings.NewReplacer("{{HD4}}", itoa(hd/4), "{{KVD4}}", itoa(kvd/4), "{{CTXKVD4}}", itoa(ctx*kvd/4), "{{CTX}}", itoa(ctx),
		"{{GQ}}", itoa(heads/kvheads), "{{G}}", itoa(wg/(hd/4)), "{{S}}", itoa((ctx+wg-1)/wg), "{{SCALE}}", f32lit(scale), "{{WG}}", itoa(wg)).Replace(fusedAttn)
}

// fusedShaders returns the fused kernels for a sample model (SmolLM2-135M's
// dimensions) at workgroup size wg, for tools that validate them.
func fusedShaders(wg int) map[string]string {
	c := Config{Embd: 576, FF: 1536, Heads: 9, KVHeads: 3, HeadDim: 64, Eps: 1e-5}
	g := &GPU{Cfg: c, wg: wg, ctx: 512, lim: DefaultLimits()}
	kvd := c.KVHeads * c.HeadDim
	nqkv := c.Embd + 2*kvd
	out := map[string]string{}
	add := func(n string, s mvSpec, k mvKind, extra map[string]int) {
		src, _, _ := fusedMVSource(s, k, extra)
		out[n] = src
	}
	add("k1", g.spec(c.Embd, nqkv), mvKind{mode: "qkv", norm: true}, map[string]int{
		"NQ": c.Embd, "NK": kvd, "HD": c.HeadDim, "HP": c.HeadDim / 2, "KVD": kvd, "CTXKVD": 512 * kvd, "NW": nqkv * c.Embd / 32})
	out["k2"] = fusedAttnSource(wg, c.HeadDim, c.Heads, c.KVHeads, 512, 0.125)
	add("k3", g.spec(c.Embd, c.Embd), mvKind{mode: "acc", attnIn: true}, map[string]int{"HD4": c.HeadDim / 4, "S": (512 + wg - 1) / wg})
	add("k4", g.spec(c.Embd, 2*c.FF), mvKind{mode: "gateup", norm: true}, map[string]int{"F": c.FF, "NW": 2 * c.FF * c.Embd / 32})
	add("k5_0", g.spec(c.FF, c.Embd), mvKind{mode: "acc"}, nil)
	add("k5_1", g.spec(c.FF, c.Embd), mvKind{mode: "acc", q41: true}, nil)
	out["am1"], out["am2"] = argmaxSources(wg, 49152)
	return out
}

// Greedy sampling in two stages, so that the 49152 logits are not walked by
// one workgroup: NWG workgroups each reduce a slice of the logits to its
// (largest, lowest index among equals), then one workgroup reduces the NWG
// partials and writes the token. Ties go to the lowest index, like llama.cpp.
const fusedArgmax1 = `
const VOCAB: u32 = {{VOCAB}}u;
const SL: u32 = {{SL}}u;
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> pbv: array<f32>;
@group(0) @binding(2) var<storage, read_write> pbi: array<u32>;
var<workgroup> bv: array<f32, {{WG}}>;
var<workgroup> bi: array<u32, {{WG}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lo = wid.x * SL;
  let hi = min(lo + SL, VOCAB);
  var best = -3.0e38;
  var bidx = lo;
  var i = lo + li;
  loop {
    if (i >= hi) { break; }
    let v = logits[i];
    if (v > best) { best = v; bidx = i; }
    i = i + {{WG}}u;
  }
  bv[li] = best;
  bi[li] = bidx;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) {
      let ov = bv[li + s];
      let oi = bi[li + s];
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (li == 0u) { pbv[wid.x] = bv[0]; pbi[wid.x] = bi[0]; }
}
`

const fusedArgmax2 = `
const NWG: u32 = {{NWG}}u;
@group(0) @binding(0) var<storage, read> pbv: array<f32>;
@group(0) @binding(1) var<storage, read> pbi: array<u32>;
@group(0) @binding(2) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> outv: array<u32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
var<workgroup> bv: array<f32, {{WG}}>;
var<workgroup> bi: array<u32, {{WG}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_index) li: u32) {
  var best = -3.0e38;
  var bidx = 0xffffffffu;
  var i = li;
  loop {
    if (i >= NWG) { break; }
    let v = pbv[i];
    if (v > best) { best = v; bidx = pbi[i]; }
    i = i + {{WG}}u;
  }
  bv[li] = best;
  bi[li] = bidx;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) {
      let ov = bv[li + s];
      let oi = bi[li + s];
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (li == 0u) {
    tokens[state[1] + 1u] = bi[0];
    outv[0] = bi[0];
  }
}
`

// argmaxGroups is NWG, the workgroups of the first stage.
const argmaxGroups = 64

func argmaxSources(wg, vocab int) (a1, a2 string) {
	sl := (vocab + argmaxGroups - 1) / argmaxGroups
	r := strings.NewReplacer("{{VOCAB}}", itoa(vocab), "{{SL}}", itoa(sl), "{{WG}}", itoa(wg), "{{NWG}}", itoa(argmaxGroups))
	return r.Replace(fusedArgmax1), r.Replace(fusedArgmax2)
}
