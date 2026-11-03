package llm

import (
	"slices"
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
