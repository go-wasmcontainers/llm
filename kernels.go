package llm

import "strings"

// WGSL kernels of the f32 path. Nothing here depends on a vendor: no
// subgroups, no f16, workgroup memory of WG*4 bytes, and no assumption on
// the size of a hardware wave. {{WG}} is the workgroup size, chosen from the
// adapter's limits (Limits.workgroupSize); it is a power of two.
//
// Reductions are tree reductions in workgroup memory, so their summation
// order depends on WG and on the number of elements, never on the hardware:
// two adapters with the same WG sum in the same order (fused multiply-adds
// aside, which the compiler may introduce).

const wgslPrelude = `
var<workgroup> red: array<f32, {{WG}}>;

fn rsum(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

fn rmax(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
    s = s >> 1u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`

// Binding access strings: 'r' read-only storage, 'w' read-write storage.

const kernelEmbed = `
struct Meta { n: u32 }
@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> dd: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
@group(0) @binding(5) var<storage, read> prm: Meta;

// Q8_0 row gather: element e of the token's row is d * int8.
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_index) li: u32) {
  let tok = tokens[state[1]];
  let nb = prm.n / 32u;
  var e = li;
  loop {
    if (e >= prm.n) { break; }
    let b = e / 32u;
    let w = qs[(tok * nb + b) * 8u + (e % 32u) / 4u];
    let v = extractBits(bitcast<i32>(w), 8u * (e % 4u), 8u);
    x[e] = dd[tok * nb + b] * f32(v);
    e = e + {{WG}}u;
  }
}
`

const kernelRMS = prelude + `
struct Meta { n: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> prm: Meta;

@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_index) li: u32) {
  var ss = 0.0;
  var i = li;
  loop {
    if (i >= prm.n) { break; }
    ss = ss + x[i] * x[i];
    i = i + {{WG}}u;
  }
  let tot = rsum(li, ss);
  let s = 1.0 / sqrt(tot / f32(prm.n) + prm.eps);
  i = li;
  loop {
    if (i >= prm.n) { break; }
    y[i] = x[i] * s * w[i];
    i = i + {{WG}}u;
  }
}
`

// matvec kernels: one workgroup per output row; the threads of the group
// stride over the 32-bit words of the row; tree reduction. Meta: nin, nout,
// acc (1: y += result), nx (workgroups in x of the 2-D grid).

const matvecHead = prelude + `
struct Meta { nin: u32, nout: u32, acc: u32, nx: u32 }
`

const kernelMatvecQ40 = matvecHead + `
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> dd: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> prm: Meta;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.y * prm.nx + wid.x;
  let valid = row < prm.nout;
  let r = select(0u, row, valid);
  let nb = prm.nin / 32u;
  let words = nb * 4u;
  var acc = 0.0;
  var w = li;
  loop {
    if (w >= words) { break; }
    let b = w / 4u;
    let k = w % 4u;
    let word = qs[r * words + w];
    let sc = dd[r * nb + b];
    let xb = b * 32u + 4u * k;
    var s = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
      let bt = (word >> (8u * j)) & 0xffu;
      s = s + f32(i32(bt & 15u) - 8) * x[xb + j] + f32(i32(bt >> 4u) - 8) * x[xb + 16u + j];
    }
    acc = acc + sc * s;
    w = w + {{WG}}u;
  }
  let tot = rsum(li, acc);
  if (li == 0u && valid) {
    if (prm.acc == 1u) { y[row] = y[row] + tot; } else { y[row] = tot; }
  }
}
`

const kernelMatvecQ41 = matvecHead + `
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> dd: array<f32>;
@group(0) @binding(2) var<storage, read> mm: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> prm: Meta;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.y * prm.nx + wid.x;
  let valid = row < prm.nout;
  let r = select(0u, row, valid);
  let nb = prm.nin / 32u;
  let words = nb * 4u;
  var acc = 0.0;
  var w = li;
  loop {
    if (w >= words) { break; }
    let b = w / 4u;
    let k = w % 4u;
    let word = qs[r * words + w];
    let xb = b * 32u + 4u * k;
    var s = 0.0;
    var sx = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
      let bt = (word >> (8u * j)) & 0xffu;
      let x0 = x[xb + j];
      let x1 = x[xb + 16u + j];
      s = s + f32(bt & 15u) * x0 + f32(bt >> 4u) * x1;
      sx = sx + x0 + x1;
    }
    acc = acc + dd[r * nb + b] * s + mm[r * nb + b] * sx;
    w = w + {{WG}}u;
  }
  let tot = rsum(li, acc);
  if (li == 0u && valid) {
    if (prm.acc == 1u) { y[row] = y[row] + tot; } else { y[row] = tot; }
  }
}
`

const kernelMatvecQ80 = matvecHead + `
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> dd: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> prm: Meta;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.y * prm.nx + wid.x;
  let valid = row < prm.nout;
  let r = select(0u, row, valid);
  let nb = prm.nin / 32u;
  let words = nb * 8u;
  var acc = 0.0;
  var w = li;
  loop {
    if (w >= words) { break; }
    let b = w / 8u;
    let k = w % 8u;
    let word = bitcast<i32>(qs[r * words + w]);
    let xb = b * 32u + 4u * k;
    var s = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
      s = s + f32(extractBits(word, 8u * j, 8u)) * x[xb + j];
    }
    acc = acc + dd[r * nb + b] * s;
    w = w + {{WG}}u;
  }
  let tot = rsum(li, acc);
  if (li == 0u && valid) {
    if (prm.acc == 1u) { y[row] = y[row] + tot; } else { y[row] = tot; }
  }
}
`

// RoPE on adjacent pairs (llama.cpp rope type NORM) from a cos/sin table
// computed on the CPU in double precision (so no adapter's sin/cos
// accuracy enters), then K and V are appended to the cache at the position.
// Meta: heads, kvHeads, headDim, ctx. cs[(pos*(hd/2)+i)*2 +{0,1}] = cos, sin.
const kernelRopeKV = `
struct Meta { heads: u32, kvheads: u32, hd: u32, ctx: u32 }
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> kin: array<f32>;
@group(0) @binding(2) var<storage, read> vin: array<f32>;
@group(0) @binding(3) var<storage, read_write> kc: array<f32>;
@group(0) @binding(4) var<storage, read_write> vc: array<f32>;
@group(0) @binding(5) var<storage, read> state: array<u32>;
@group(0) @binding(6) var<storage, read> cs: array<f32>;
@group(0) @binding(7) var<storage, read> prm: Meta;

@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * {{WG}}u) + gid.x;
  let hp = prm.hd / 2u;
  let nq = prm.heads * hp;
  let nk = prm.kvheads * hp;
  let kvd = prm.kvheads * prm.hd;
  let pos = state[0];
  if (i < nq) {
    let h = i / hp;
    let p = i % hp;
    let c = cs[(pos * hp + p) * 2u];
    let s = cs[(pos * hp + p) * 2u + 1u];
    let a = q[h * prm.hd + 2u * p];
    let b = q[h * prm.hd + 2u * p + 1u];
    q[h * prm.hd + 2u * p] = a * c - b * s;
    q[h * prm.hd + 2u * p + 1u] = a * s + b * c;
  } else if (i < nq + nk) {
    let j = i - nq;
    let h = j / hp;
    let p = j % hp;
    let c = cs[(pos * hp + p) * 2u];
    let s = cs[(pos * hp + p) * 2u + 1u];
    let a = kin[h * prm.hd + 2u * p];
    let b = kin[h * prm.hd + 2u * p + 1u];
    kc[pos * kvd + h * prm.hd + 2u * p] = a * c - b * s;
    kc[pos * kvd + h * prm.hd + 2u * p + 1u] = a * s + b * c;
  } else if (i < nq + nk + kvd) {
    let j = i - nq - nk;
    vc[pos * kvd + j] = vin[j];
  }
}
`

// Attention of one query position against the cache, one workgroup per
// head: scores into a scratch buffer, softmax with a max subtraction, then
// the weighted sum of V. Meta: heads, kvHeads, headDim, ctx, scale.
const kernelAttn = prelude + `
struct Meta { heads: u32, kvheads: u32, hd: u32, ctx: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;
@group(0) @binding(5) var<storage, read> state: array<u32>;
@group(0) @binding(6) var<storage, read> prm: Meta;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wid.x;
  let kvh = h / (prm.heads / prm.kvheads);
  let n = state[0] + 1u;
  let kvd = prm.kvheads * prm.hd;
  let sb = h * prm.ctx;
  var mx = -3.0e38;
  var t = li;
  loop {
    if (t >= n) { break; }
    var s = 0.0;
    for (var d = 0u; d < prm.hd; d = d + 1u) {
      s = s + q[h * prm.hd + d] * kc[t * kvd + kvh * prm.hd + d];
    }
    s = s * prm.scale;
    sc[sb + t] = s;
    mx = max(mx, s);
    t = t + {{WG}}u;
  }
  let gmax = rmax(li, mx);
  var sum = 0.0;
  t = li;
  loop {
    if (t >= n) { break; }
    let e = exp(sc[sb + t] - gmax);
    sc[sb + t] = e;
    sum = sum + e;
    t = t + {{WG}}u;
  }
  let gsum = rsum(li, sum);
  storageBarrier();
  workgroupBarrier();
  var oc = li;
  loop {
    if (oc >= prm.hd) { break; }
    var o = 0.0;
    for (var u = 0u; u < n; u = u + 1u) {
      o = o + sc[sb + u] * vc[u * kvd + kvh * prm.hd + oc];
    }
    outv[h * prm.hd + oc] = o / gsum;
    oc = oc + {{WG}}u;
  }
}
`

// SwiGLU in place: g = silu(g) * u. Meta: n.
const kernelSwiGLU = `
struct Meta { n: u32 }
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> prm: Meta;

@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < prm.n) {
    let v = g[i];
    g[i] = v / (1.0 + exp(-v)) * u[i];
  }
}
`

// Greedy sampling on the GPU: the index of the largest logit (the lowest on
// ties, like llama.cpp's greedy sampler) goes to tokens[step+1] and to out[0],
// the only four bytes the guest reads back. Meta: n.
const kernelArgmax = `
struct Meta { n: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> outv: array<u32>;
@group(0) @binding(3) var<storage, read> state: array<u32>;
@group(0) @binding(4) var<storage, read> prm: Meta;
var<workgroup> bv: array<f32, {{WG}}>;
var<workgroup> bi: array<u32, {{WG}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_index) li: u32) {
  var best = -3.0e38;
  var bidx = 0u;
  var i = li;
  loop {
    if (i >= prm.n) { break; }
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
  if (li == 0u) {
    tokens[state[1] + 1u] = bi[0];
    outv[0] = bi[0];
  }
}
`

// Advance the position and the step after a token.
const kernelAdvance = `
@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@compute @workgroup_size(1)
fn main() {
  state[0] = state[0] + 1u;
  state[1] = state[1] + 1u;
}
`

const prelude = wgslPrelude

func wgsl(src string, wg int) string {
	return strings.ReplaceAll(src, "{{WG}}", itoa(wg))
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// Shaders returns every kernel's WGSL for workgroup size wg, by name: for
// tools that validate them (cmd/wgsldump).
func Shaders(wg int) map[string]string {
	m := fusedShaders(wg)
	for k, v := range map[string]string{
		"embed": wgsl(kernelEmbed, wg), "rms": wgsl(kernelRMS, wg), "q40": wgsl(kernelMatvecQ40, wg), "q41": wgsl(kernelMatvecQ41, wg),
		"q80": wgsl(kernelMatvecQ80, wg), "rope": wgsl(kernelRopeKV, wg), "attn": wgsl(kernelAttn, wg), "swi": wgsl(kernelSwiGLU, wg),
		"argmax": wgsl(kernelArgmax, wg), "adv": wgsl(kernelAdvance, wg),
	} {
		m[k] = v
	}
	return m
}
