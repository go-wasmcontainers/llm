package llm

import (
	"strconv"
	"strings"

	"github.com/go-gfx/webgpu/kernels/argmax"
	"github.com/go-gfx/webgpu/kernels/q4"
)

// Fused decode kernels. One layer of the pass is five dispatches:
//
//	K1  RMSNorm + Q,K,V matrix products + RoPE + KV-cache append
//	K2  attention (all heads, in chunks of WG positions)
//	K3  merge of the chunks + attention output projection, added to the residual
//	K4  RMSNorm + gate and up products + SiLU(gate)*up
//	K5  down projection, added to the residual
//
// The matrix-vector kernel itself (Q4_0 / Q4_1 blocks, staged input, TPR
// invocations per row, tiling by the adapter's workgroup memory, the token on
// the z axis) is github.com/go-gfx/webgpu/kernels/q4; what is model-specific
// (RoPE, SwiGLU, the chunk merge, attention) is here, passed as q4.Hooks.

const wgslPrelude = q4.Prelude

// Gate/up: a workgroup computes RPW/2 rows of gate and the same rows of up
// (the matrices are concatenated [gate; up], F rows each).
const fusedRowGateUp = `
  let hh = RPW / 2u;
  let orow = wg * hh + lr % hh;
  let valid = orow < F;
  let row = select(orow, F + orow, lr >= hh);`

const fusedEpiGateUp = `
  if (li < RPW / 2u) {
    let o = wg * (RPW / 2u) + li;
    if (o < F) {
      let g = red[li * TPR];
      let u = red[(li + RPW / 2u) * TPR];
      act[tk * F + o] = g / (1.0 + exp(-g)) * u;
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
      let pos = state[0] + tk;
      if (rr < NQ + NK) {
        var j = rr;
        if (rr >= NQ) { j = rr - NQ; }
        let p = (j % HD) / 2u;
        let c = cs[(pos * HP + p) * 2u];
        let sn = cs[(pos * HP + p) * 2u + 1u];
        let ra = a * c - b * sn;
        let rb = a * sn + b * c;
        if (rr < NQ) {
          q[tk * NQ + rr] = ra; q[tk * NQ + rr + 1u] = rb;
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

// Attention, all heads of one query position. One workgroup per (chunk of WG
// positions, head): scores (one position per invocation, vec4 loads), the
// partial softmax of the chunk (its max m and sum l of exp(s-m)), then the
// weighted sum of V with the chunk split over G = WG/HD4 groups of HD4
// invocations (one vec4 of the head per invocation). K3 merges the chunks of
// a head (flash-decoding: Dao et al.), so no loop of an invocation is longer
// than HD4.
const fusedAttn = wgslPrelude + `
const HD4: u32 = {{HD4}}u;
const KVD4: u32 = {{KVD4}}u;
const CTXKVD4: u32 = {{CTXKVD4}}u;
const CTX: u32 = {{CTX}}u;
const GQ: u32 = {{GQ}}u;
const G: u32 = {{G}}u;
const WG: u32 = {{WG}}u;
const S: u32 = {{S}}u;
const H: u32 = {{H}}u;
const SCALE: f32 = {{SCALE}};
@group(0) @binding(0) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pv: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ml: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
var<workgroup> qh: array<vec4<f32>, {{HD4}}>;
var<workgroup> pw: array<f32, {{WG}}>;
var<workgroup> part: array<vec4<f32>, {{WG}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = wid.x;
  let h = wid.y;
  let tk = wid.z;
  let kvh = h / GQ;
  let n = state[0] + tk + 1u;
  let p = c * WG + li;
  let th = tk * H + h;
  if (li < HD4) { qh[li] = q4[th * HD4 + li]; }
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
    pv[(th * S + c) * HD4 + li] = tot;
  }
  if (li == 0u) { ml[th * S + c] = vec2<f32>(m, l); }
}
`

// The merge of the chunks of a head, as the input stage of K3: element gi (a
// vec4 of the concatenated heads) is sum_c exp(m_c-M) pv_c / sum_c exp(m_c-M) l_c.
const fusedCombine = `
fn combine(gi: u32, tk: u32) -> vec4<f32> {
  let h = tk * (N / 4u / HD4) + gi / HD4;
  let d4 = gi % HD4;
  let sn = (state[0] + tk + WG) / WG;
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

func f32lit(x float32) string { return q4.F32(x) }

// mvKind: mode is "qkv", "gateup", "acc" or "store"; norm adds the RMSNorm
// (weights at aux[NW:]) in front; attnIn takes the merge of the attention
// chunks as the input.
type mvKind struct {
	q41    bool
	mode   string
	norm   bool
	attnIn bool
}

// fusedMVSource returns the WGSL of a fused matrix-vector kernel, with extra
// constants for the epilogue, its binding access string and its number of
// workgroups.
func fusedMVSource(s mvSpec, k mvKind, extra map[string]int) (src, access string, nwgs int) {
	sp := q4.Spec{WG: s.wg, TPR: s.tpr, MaxShared: s.maxShared, MaxGroups: s.maxGroups, N: s.n, Rows: s.rows, Eps: s.eps,
		Q41: k.q41, Norm: k.norm, NW: extra["NW"]}
	var h q4.Hooks
	consts := func(names ...string) string {
		var d string
		for _, n := range names {
			d += "const " + n + ": u32 = " + strconv.Itoa(extra[n]) + "u;\n"
		}
		return d
	}
	groups := 0
	switch k.mode {
	case "acc":
		sp.Mode = q4.Acc
	case "store":
		sp.Mode = q4.Store
	case "qkv":
		sp.Mode = q4.Custom
		h.Bindings = []q4.Binding{{Name: "q", Type: "array<f32>", RW: true}, {Name: "kv", Type: "array<f32>", RW: true},
			{Name: "state", Type: "array<u32>"}, {Name: "cs", Type: "array<f32>"}}
		h.Decl = consts("NQ", "NK", "HD", "HP", "KVD", "CTXKVD")
		h.RowMap, h.Epi = q4.RowPlain, fusedEpiQKV
	case "gateup":
		sp.Mode = q4.Custom
		h.Bindings = []q4.Binding{{Name: "act", Type: "array<f32>", RW: true}}
		h.Decl = consts("F")
		h.RowMap, h.Epi = fusedRowGateUp, fusedEpiGateUp
		half := s.wg / s.tpr / 2
		groups = (extra["F"] + half - 1) / half
	}
	if k.attnIn {
		h.NoInput = true
		h.Bindings = append([]q4.Binding{{Name: "pv", Type: "array<vec4<f32>>"}, {Name: "ml", Type: "array<vec2<f32>>"}, {Name: "state", Type: "array<u32>"}}, h.Bindings...)
		h.Decl += consts("HD4", "S")
		h.Func = fusedCombine
		h.Load = "combine(gi, tk)"
	}
	sp.Hooks = h
	src, access, gx, gy := q4.Source(sp, groups)
	return src, access, gx * gy
}

func fusedAttnSource(wg, hd, heads, kvheads, ctx int, scale float32) string {
	kvd := kvheads * hd
	return strings.NewReplacer("{{HD4}}", itoa(hd/4), "{{KVD4}}", itoa(kvd/4), "{{CTXKVD4}}", itoa(ctx*kvd/4), "{{CTX}}", itoa(ctx),
		"{{GQ}}", itoa(heads/kvheads), "{{G}}", itoa(wg/(hd/4)), "{{S}}", itoa((ctx+wg-1)/wg), "{{H}}", itoa(heads), "{{SCALE}}", f32lit(scale),
		"{{WG}}", itoa(wg)).Replace(fusedAttn)
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
	out["am1"], out["am2"] = argmax.Sources(wg, 49152)
	return out
}

// argmaxGroups is the workgroups of the first argmax stage.
const argmaxGroups = argmax.Groups

func argmaxSources(wg, vocab int) (a1, a2 string) { return argmax.Sources(wg, vocab) }
