package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// build makes a tiny valid file: one metadata string, one F32 tensor of 4.
func build(mut func(b *bytes.Buffer)) []byte {
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(1))
	w(uint64(2))
	str("general.architecture")
	w(uint32(String))
	str("llama")
	str("x.count")
	w(uint32(U32))
	w(uint32(7))
	str("w")
	w(uint32(1))
	w(uint64(4))
	w(uint32(TypeF32))
	w(uint64(0))
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, 16))
	if mut != nil {
		mut(&b)
	}
	return b.Bytes()
}

func TestParse(t *testing.T) {
	d := build(nil)
	f, err := Parse(bytes.NewReader(d), uint64(len(d)))
	if err != nil {
		t.Fatal(err)
	}
	if s, _ := f.Str("general.architecture"); s != "llama" {
		t.Fatal(s)
	}
	if n, _ := f.Uint("x.count"); n != 7 {
		t.Fatal(n)
	}
	w, ok := f.Tensor("w")
	if !ok || w.Size != 16 || f.DataOffset%32 != 0 {
		t.Fatalf("%+v %d", w, f.DataOffset)
	}
}

func TestRefusals(t *testing.T) {
	d := build(nil)
	cases := map[string]struct {
		data []byte
		want error
	}{
		"magic":     {append([]byte("GGUX"), d[4:]...), ErrMagic},
		"version":   {func() []byte { c := bytes.Clone(d); c[4] = 2; return c }(), ErrVersion},
		"tensors":   {func() []byte { c := bytes.Clone(d); c[8+7] = 0x7f; return c }(), ErrBound},
		"truncated": {d[:40], nil},
		"past end":  {d[:len(d)-4], ErrBound},
	}
	for n, c := range cases {
		_, err := Parse(bytes.NewReader(c.data), uint64(len(c.data)))
		if err == nil || c.want != nil && !errors.Is(err, c.want) {
			t.Errorf("%s: %v", n, err)
		}
	}
}

func FuzzParse(f *testing.F) {
	f.Add(build(nil))
	f.Fuzz(func(t *testing.T, d []byte) {
		Parse(bytes.NewReader(d), uint64(len(d)))
	})
}

// TestModel reads the header of the pinned SmolLM2 file when LLM_MODELS is set.
func TestModel(t *testing.T) {
	dir := os.Getenv("LLM_MODELS")
	if dir == "" {
		t.Skip("LLM_MODELS not set")
	}
	fh, err := os.Open(filepath.Join(dir, "SmolLM2-135M-Instruct-Q4_0.gguf"))
	if err != nil {
		t.Skip(err)
	}
	defer fh.Close()
	st, _ := fh.Stat()
	f, err := Parse(fh, uint64(st.Size()))
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range f.Meta {
		if a, ok := v.([]any); ok {
			t.Logf("%s = array(%d)", k, len(a))
		} else {
			t.Logf("%s = %v", k, v)
		}
	}
	for _, x := range f.Tensors[:12] {
		t.Logf("%s %v type %d off %d size %d", x.Name, x.Dims, x.Type, x.Offset, x.Size)
	}
	t.Logf("tensors %d", len(f.Tensors))
}

func TestModelTypes(t *testing.T) {
	dir := os.Getenv("LLM_MODELS")
	if dir == "" {
		t.Skip("LLM_MODELS not set")
	}
	fh, err := os.Open(filepath.Join(dir, "SmolLM2-135M-Instruct-Q4_0.gguf"))
	if err != nil {
		t.Skip(err)
	}
	defer fh.Close()
	st, _ := fh.Stat()
	f, _ := Parse(fh, uint64(st.Size()))
	n := map[string]int{}
	for _, x := range f.Tensors {
		k := "other"
		if len(x.Dims) == 2 {
			k = "mat"
		}
		n[k+" type "+string(rune('0'+x.Type/10))+string(rune('0'+x.Type%10))+" "+tensorKind(x.Name)]++
	}
	t.Log(n)
	_, ok := f.Tensor("output.weight")
	t.Log("output.weight present:", ok)
}

func tensorKind(n string) string {
	for i := len(n) - 1; i >= 0; i-- {
		if n[i] == '.' && i > 0 {
			j := i - 1
			for j >= 0 && n[j] != '.' {
				j--
			}
			return n[j+1 : i]
		}
	}
	return n
}
