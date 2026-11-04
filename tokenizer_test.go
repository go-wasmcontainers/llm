package llm

import (
	"slices"
	"strings"
	"testing"
)

func TestSplitGPT2(t *testing.T) {
	for in, want := range map[string][]string{
		"Hello world": {"Hello", " world"},
		"it's 5":      {"it", "'s", " 5"},
		"a  b":        {"a", " ", " b"},
		"x\n\ny":      {"x", "\n", "\n", "y"},
		"end  ":       {"end", "  "},
		"foo(bar):":   {"foo", "(", "bar", "):"},
		"dé jà":       {"dé", " jà"},
	} {
		if got := splitGPT2(in); !slices.Equal(got, want) {
			t.Errorf("%q: %q, want %q", in, got, want)
		}
	}
}

func TestSplitDigits(t *testing.T) {
	if got := splitDigits("ab12c"); !slices.Equal(got, []string{"ab", "1", "2", "c"}) {
		t.Fatalf("%q", got)
	}
}

func TestByteMap(t *testing.T) {
	if byteToRune[' '] != 0x120 || byteToRune['\n'] != 0x10a || byteToRune['A'] != 'A' {
		t.Fatalf("%x %x", byteToRune[' '], byteToRune['\n'])
	}
	for b := 0; b < 256; b++ {
		if runeToByte[byteToRune[b]] != byte(b) {
			t.Fatal(b)
		}
	}
}

func TestF16RoundTrip(t *testing.T) {
	for _, v := range []float32{0, 1, -2, 0.5, 65504, 6.1e-5, 5.96e-8, 3.14159} {
		h := F32ToF16(v)
		if back := F16ToF32(h); F32ToF16(back) != h {
			t.Errorf("%v: %x -> %v", v, h, back)
		}
	}
	if F32ToF16(1) != 0x3c00 || F16ToF32(0x3c00) != 1 || F32ToF16(65520) != 0x7c00 {
		t.Fatal("known values")
	}
}

// The qwen2 pre-tokenizer on cases worked out from llama.cpp's regex
// (llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2); the browser-free oracle test
// TestOracleQwenDequantizedF32 compares the token ids with llama.cpp's.
func TestSplitQwen2(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"It's DON'T", []string{"It", "'s", " DON", "'T"}},
		{"a\n\n  b", []string{"a", "\n\n", " ", " b"}},
		{"x = 12", []string{"x", " =", " ", "1", "2"}},
		{"end ", []string{"end", " "}},
		{"$$ hi", []string{"$$", " hi"}},
		{"line\r\n\r\nnext", []string{"line", "\r\n\r\n", "next"}},
		{"", nil},
	} {
		got := splitQwen2(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("%q: %q, want %q", c.in, got, c.want)
		}
	}
}
