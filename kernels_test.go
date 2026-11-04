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
