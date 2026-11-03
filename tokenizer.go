package llm

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/go-wasmcontainers/llm/gguf"
)

// Tokenizer is the byte-level BPE ("gpt2") tokenizer stored in GGUF
// metadata (tokenizer.ggml.tokens, .merges, .token_type), with the
// "smollm" pre-tokenizer of llama.cpp's llama-vocab.cpp: digits split one by
// one, then the GPT-2 pattern
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S).
type Tokenizer struct {
	tokens  []string
	ids     map[string]int32
	ranks   map[[2]string]int
	special []string // control and user-defined tokens, longest first
	EOS     int32
	BOS     int32
	AddBOS  bool
}

var byteToRune [256]rune
var runeToByte = map[rune]byte{}

func init() {
	// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest
	// to U+0100 and up in order.
	var bs []int
	for b := '!'; b <= '~'; b++ {
		bs = append(bs, int(b))
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}
	has := map[int]bool{}
	for _, b := range bs {
		has[b] = true
		byteToRune[b] = rune(b)
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !has[b] {
			byteToRune[b] = rune(256 + n)
			n++
		}
	}
	for b := 0; b < 256; b++ {
		runeToByte[byteToRune[b]] = byte(b)
	}
}

// NewTokenizer builds the tokenizer from a parsed header.
func NewTokenizer(f *gguf.File) (*Tokenizer, error) {
	if m, _ := f.Str("tokenizer.ggml.model"); m != "gpt2" {
		return nil, fmt.Errorf("llm: tokenizer model %q is not supported (gpt2 only)", m)
	}
	if p, _ := f.Str("tokenizer.ggml.pre"); p != "smollm" {
		return nil, fmt.Errorf("llm: pre-tokenizer %q is not supported (smollm only)", p)
	}
	toks, _ := f.Meta["tokenizer.ggml.tokens"].([]any)
	merges, _ := f.Meta["tokenizer.ggml.merges"].([]any)
	types, _ := f.Meta["tokenizer.ggml.token_type"].([]any)
	if len(toks) == 0 {
		return nil, errors.New("llm: no tokenizer.ggml.tokens")
	}
	t := &Tokenizer{ids: make(map[string]int32, len(toks)), ranks: make(map[[2]string]int, len(merges))}
	for i, v := range toks {
		s, ok := v.(string)
		if !ok {
			return nil, errors.New("llm: token is not a string")
		}
		t.tokens = append(t.tokens, s)
		t.ids[s] = int32(i)
		if i < len(types) {
			if ty, _ := types[i].(int64); ty == 3 || ty == 4 { // control, user defined
				t.special = append(t.special, s)
			}
		}
	}
	sort.Slice(t.special, func(i, j int) bool { return len(t.special[i]) > len(t.special[j]) })
	for i, v := range merges {
		s, _ := v.(string)
		a, b, ok := strings.Cut(s, " ")
		if !ok {
			return nil, fmt.Errorf("llm: merge %d %q", i, s)
		}
		t.ranks[[2]string{a, b}] = i
	}
	if v, ok := f.Uint("tokenizer.ggml.eos_token_id"); ok {
		t.EOS = int32(v)
	}
	if v, ok := f.Uint("tokenizer.ggml.bos_token_id"); ok {
		t.BOS = int32(v)
	}
	t.AddBOS, _ = f.Meta["tokenizer.ggml.add_bos_token"].(bool)
	return t, nil
}

// VocabSize is the number of tokens.
func (t *Tokenizer) VocabSize() int { return len(t.tokens) }

// Encode tokenizes text. Special tokens written in the text are matched as
// such (llama.cpp's parse_special).
func (t *Tokenizer) Encode(text string) []int32 {
	var out []int32
	if t.AddBOS {
		out = append(out, t.BOS)
	}
	for len(text) > 0 {
		// next special token occurrence
		at, which := -1, ""
		for _, s := range t.special {
			if i := strings.Index(text, s); i >= 0 && (at < 0 || i < at) {
				at, which = i, s
			}
		}
		if at < 0 {
			return t.encodePlain(out, text)
		}
		out = t.encodePlain(out, text[:at])
		out = append(out, t.ids[which])
		text = text[at+len(which):]
	}
	return out
}

func (t *Tokenizer) encodePlain(out []int32, text string) []int32 {
	for _, seg := range splitDigits(text) {
		for _, piece := range splitGPT2(seg) {
			out = t.bpe(out, piece)
		}
	}
	return out
}

// splitDigits is the pattern \p{N}: every number character on its own.
func splitDigits(s string) []string {
	var out []string
	start := 0
	for i, r := range s {
		if unicode.IsNumber(r) {
			if start < i {
				out = append(out, s[start:i])
			}
			out = append(out, s[i:i+utf8.RuneLen(r)])
			start = i + utf8.RuneLen(r)
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func isSpace(r rune) bool { return unicode.IsSpace(r) }
func isOther(r rune) bool { return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) }

// splitGPT2 implements the second pattern by hand (Go's regexp has no
// look-ahead). Text no alternative matches is emitted a character at a time.
func splitGPT2(s string) []string {
	rs := []rune(s)
	var out []string
	i := 0
	for i < len(rs) {
		start := i
		r := rs[i]
		switch {
		case r == '\'' && i+1 < len(rs):
			n := 0
			for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
				if i+1+len(c) <= len(rs) && string(rs[i+1:i+1+len(c)]) == c {
					n = 1 + len(c)
					break
				}
			}
			if n > 0 {
				i += n
				out = append(out, string(rs[start:i]))
				continue
			}
		}
		j := i
		if rs[j] == ' ' && j+1 < len(rs) {
			j++
		}
		switch {
		case unicode.IsLetter(rs[j]):
			for j < len(rs) && unicode.IsLetter(rs[j]) {
				j++
			}
			i = j
		case unicode.IsNumber(rs[j]):
			for j < len(rs) && unicode.IsNumber(rs[j]) {
				j++
			}
			i = j
		case isOther(rs[j]):
			for j < len(rs) && isOther(rs[j]) {
				j++
			}
			i = j
		case isSpace(r):
			// \s+(?!\S): the run, less its last character when a
			// non-space follows.
			k := i
			for k < len(rs) && isSpace(rs[k]) {
				k++
			}
			if k < len(rs) {
				k--
			}
			if k > i {
				i = k
			} else {
				i++ // no alternative matches: one character
			}
		default:
			i++
		}
		out = append(out, string(rs[start:i]))
	}
	return out
}

// bpe merges the bytes of piece by merge rank and appends the token ids.
func (t *Tokenizer) bpe(out []int32, piece string) []int32 {
	var syms []string
	for _, b := range []byte(piece) {
		syms = append(syms, string(byteToRune[b]))
	}
	for len(syms) > 1 {
		best, bi := int(^uint(0)>>1), -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && r < best {
				best, bi = r, i
			}
		}
		if bi < 0 {
			break
		}
		syms[bi] += syms[bi+1]
		syms = append(syms[:bi+1], syms[bi+2:]...)
	}
	for _, s := range syms {
		if id, ok := t.ids[s]; ok {
			out = append(out, id)
			continue
		}
		// unknown merged symbol: fall back to single bytes
		for _, r := range s {
			if id, ok := t.ids[string(r)]; ok {
				out = append(out, id)
			}
		}
	}
	return out
}

// Decode returns the text of tokens. Special tokens are written as they are.
func (t *Tokenizer) Decode(ids []int32) string {
	var b []byte
	for _, id := range ids {
		b = append(b, t.Piece(id)...)
	}
	return string(b)
}

// Piece is the bytes of one token.
func (t *Tokenizer) Piece(id int32) []byte {
	if id < 0 || int(id) >= len(t.tokens) {
		return nil
	}
	s := t.tokens[id]
	var b []byte
	for _, r := range s {
		if c, ok := runeToByte[r]; ok {
			b = append(b, c)
		} else {
			b = utf8.AppendRune(b, r)
		}
	}
	return b
}
