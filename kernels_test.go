package llm

import (
	"strings"
	"testing"
)

// Every placeholder of every kernel is substituted, at the workgroup sizes an
// adapter can give (the browser compiles them: go-wasmcontainers/gpu e2e
// TestLLMShaders; the matvec and argmax generators are tested in
// go-gfx/webgpu kernels/q4 and kernels/argmax).
func TestShadersFullySubstituted(t *testing.T) {
	for _, wg := range []int{64, 32, 16} {
		for name, src := range Shaders(wg) {
			if i := strings.Index(src, "{{"); i >= 0 {
				t.Errorf("wg %d kernel %s: unsubstituted placeholder at %q", wg, name, src[i:min(len(src), i+20)])
			}
		}
	}
}

// The GPU rotates adjacent pairs of Q and K rows reordered by neoxMap: that
// must equal the NEOX rotation (pairs i, i+hd/2) of the original order.
func TestNeoxRowsEqualNeoxRope(t *testing.T) {
	const hd, heads = 64, 3
	x := make([]float32, hd*heads)
	for i := range x {
		x[i] = float32((i*7919)%101)/17 - 3
	}
	perm := func(v []float32) []float32 {
		out := make([]float32, len(v))
		for nr := range out {
			out[nr] = v[neoxMap(nr, hd)]
		}
		return out
	}
	seen := map[int]bool{}
	for nr := 0; nr < hd*heads; nr++ {
		seen[neoxMap(nr, hd)] = true
	}
	if len(seen) != hd*heads {
		t.Fatalf("neoxMap is not a permutation: %d distinct rows", len(seen))
	}
	want := append([]float32(nil), x...)
	rope(want, 37, heads, hd, 1e6, true)
	got := perm(x)
	rope(got, 37, heads, hd, 1e6, false)
	want = perm(want)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("element %d: %v, want %v", i, got[i], want[i])
		}
	}
	b := []byte{0, 1, 2, 3, 4, 5, 6, 7}
	if r := neoxRows(b, 2, 4); len(r) != 8 || r[2] != 4 || r[4] != 2 { // rows 0,2,1,3: new row 1 is old row 2
		t.Fatalf("neoxRows %v", r)
	}
}
