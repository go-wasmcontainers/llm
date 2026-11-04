// Command ggufdump prints the tensors of a GGUF file: name, ggml type, dims.
package main

import (
	"fmt"
	"os"

	"github.com/go-wasmcontainers/llm/gguf"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: ggufdump FILE.gguf")
		os.Exit(2)
	}
	fh, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	st, _ := fh.Stat()
	f, err := gguf.Parse(fh, uint64(st.Size()))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, t := range f.Tensors {
		fmt.Printf("%s type=%d dims=%v\n", t.Name, t.Type, t.Dims)
	}
}
