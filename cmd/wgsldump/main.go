// Command wgsldump prints the WGSL kernels as JSON ({"name": "source"}), for
// validation in a browser (go-wasmcontainers/gpu e2e TestLLMShaders).
package main

import (
	"encoding/json"
	"os"
	"strconv"

	llm "github.com/go-wasmcontainers/llm"
)

func main() {
	wg := 64
	if len(os.Args) > 1 {
		if n, err := strconv.Atoi(os.Args[1]); err == nil {
			wg = n
		}
	}
	json.NewEncoder(os.Stdout).Encode(llm.Shaders(wg))
}
