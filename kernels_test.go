package llm

import (
	"strings"
	"testing"
)

// Every placeholder of every kernel is substituted, at the workgroup sizes an
// adapter can give (the browser compiles them: go-wasmcontainers/gpu e2e
// TestLLMShaders).
func TestShadersFullySubstituted(t *testing.T) {
	for _, wg := range []int{64, 32, 16} {
		for name, src := range Shaders(wg) {
			if i := strings.Index(src, "{{"); i >= 0 {
				t.Errorf("wg %d kernel %s: unsubstituted placeholder at %q", wg, name, src[i:min(len(src), i+20)])
			}
		}
	}
}

// The tile of the staged input never exceeds the adapter's workgroup memory.
func TestFusedTile(t *testing.T) {
	for _, shared := range []int{16384, 32768, 8192} {
		for _, n := range []int{576, 1536, 4864, 8192} {
			s := mvSpec{wg: 64, tpr: 8, maxShared: shared, n: n, rows: 64}
			t4, nt, tb := s.tile()
			if used := t4*16 + 64*4; used > shared {
				t.Errorf("shared %d n %d: tile uses %d bytes", shared, n, used)
			}
			if nt*tb*32 < n {
				t.Errorf("shared %d n %d: %d tiles of %d blocks do not cover %d floats", shared, n, nt, tb, n)
			}
		}
	}
}
