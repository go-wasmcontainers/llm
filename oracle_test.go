package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/go-wasmcontainers/llm/gguf"
)

const modelFile = "SmolLM2-135M-Instruct-Q4_0.gguf"

// Prompts are fixed. Greedy decoding, 32 tokens.
var oraclePrompts = []string{
	"The capital of France is",
	"def fibonacci(n):\n    ",
	"Once upon a time, in a small village near 12 mountains, there lived",
	"Les enfants jouent dans le jardin et",
}

const oracleTokens = 32

// Tolerances. They are written for another vendor's GPU as much as for this
// CPU: the f32 path sums in a different order (and, on a GPU, may fuse
// multiply-adds) than ggml's integer dot products, so only the
// ggml-exact CPU mode is held to token identity on every prompt; the f32
// path must agree on the first oracleTokens tokens for all prompts too, and
// its step-1 log-probabilities must stay within f32LogprobTol of llama.cpp's.
//
// Measured 2026-11-03 (Apple M4 Max, llama.cpp 0.8.0 CPU, one thread): the
// ggml-exact mode does NOT reproduce llama.cpp to 1e-3 (step-1 logit
// differences up to 0.4 nats), so no tighter bound is claimed; every token
// divergence was at a near-tie of llama.cpp's own distribution.
const (
	exactLogprobTol = 0.5  // nats, step-1 logit differences against llama.cpp's top-10
	f32LogprobTol   = 0.5  // same bound for the f32 path
	nearTie         = 0.25 // nats: largest llama.cpp-side gap at which a different token is accepted
	minEqual        = 6    // tokens that must agree before any divergence
)

// oracleRecord is what llama.cpp (llama-server 0.8.0, CPU, one thread, greedy)
// produced for a prompt: the data the browser test compares the GPU with.
type oracleRecord struct {
	Prompt    string     `json:"prompt"`
	PromptIDs []int32    `json:"prompt_ids"`
	Tokens    []int32    `json:"tokens"`
	Steps     [][]refTop `json:"steps"` // llama.cpp's top-10 log-probabilities at every generated token
}

type server struct {
	base string
	cmd  *exec.Cmd
}

func startServer(t *testing.T, model string, extra ...string) *server {
	bin, err := exec.LookPath("llama-server")
	if err != nil {
		t.Skip("llama-server not in PATH")
	}
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, bin, "-m", model, "-t", "1", "-ngl", "0", "-c", "512", "--port", fmt.Sprint(port), "--host", "127.0.0.1", "--no-warmup")
	cmd.Args = append(cmd.Args, extra...)
	cmd.Stdin = nil
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cancel(); cmd.Wait() })
	s := &server{base: fmt.Sprintf("http://127.0.0.1:%d", port), cmd: cmd}
	for i := 0; i < 120; i++ {
		if r, err := http.Get(s.base + "/health"); err == nil {
			ok := r.StatusCode == 200
			r.Body.Close()
			if ok {
				return s
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatal("llama-server did not become healthy")
	return nil
}

func (s *server) post(t *testing.T, path string, in, out any) {
	b, _ := json.Marshal(in)
	r, err := http.Post(s.base+path, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		t.Fatalf("%s: %s", path, r.Status)
	}
	if err := json.NewDecoder(r.Body).Decode(out); err != nil {
		t.Fatal(err)
	}
}

type refTop struct {
	ID      int32   `json:"id"`
	Logprob float64 `json:"logprob"`
}

type refResult struct {
	Tokens []int32 `json:"tokens"`
	Probs  []struct {
		Top []refTop `json:"top_logprobs"`
	} `json:"completion_probabilities"`
}

func logSoftmax(l []float32) []float64 {
	mx := float64(l[0])
	for _, v := range l {
		mx = math.Max(mx, float64(v))
	}
	var s float64
	for _, v := range l {
		s += math.Exp(float64(v) - mx)
	}
	out := make([]float64, len(l))
	for i, v := range l {
		out[i] = float64(v) - mx - math.Log(s)
	}
	return out
}

func generate(m *CPU, prompt []int32, n int) (ids []int32, step1 []float32) {
	m.Reset()
	var logits []float32
	for _, tk := range prompt {
		logits = m.Forward(tk)
	}
	step1 = logits
	for len(ids) < n {
		next := Argmax(logits)
		ids = append(ids, next)
		logits = m.Forward(next)
	}
	return
}

func TestOracleLlamaCpp(t *testing.T) {
	dir := os.Getenv("LLM_MODELS")
	if dir == "" {
		t.Skip("LLM_MODELS not set")
	}
	path := filepath.Join(dir, modelFile)
	fh, err := os.Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer fh.Close()
	st, _ := fh.Stat()
	f, err := gguf.Parse(fh, uint64(st.Size()))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewTokenizer(f)
	if err != nil {
		t.Fatal(err)
	}
	srv := startServer(t, path)
	m, err := LoadCPU(f, fh, 256)
	if err != nil {
		t.Fatal(err)
	}
	var record []oracleRecord
	defer func() {
		if out := os.Getenv("LLM_ORACLE_OUT"); out != "" && !t.Failed() {
			b, _ := json.MarshalIndent(record, "", " ")
			if err := os.WriteFile(out, append(b, '\n'), 0o644); err != nil {
				t.Error(err)
			}
		}
	}()
	for _, prompt := range oraclePrompts {
		var ref refResult
		srv.post(t, "/completion", map[string]any{"prompt": prompt, "n_predict": oracleTokens, "temperature": 0,
			"return_tokens": true, "n_probs": 10, "cache_prompt": false, "seed": 1}, &ref)
		var tk struct {
			Tokens []int32 `json:"tokens"`
		}
		srv.post(t, "/tokenize", map[string]any{"content": prompt}, &tk)
		ids := tok.Encode(prompt)
		if !slices.Equal(ids, tk.Tokens) {
			t.Errorf("%q: tokenizer\n ours %v\n ref  %v", prompt, ids, tk.Tokens)
			continue
		}
		if len(ref.Tokens) < oracleTokens || len(ref.Probs) == 0 {
			t.Fatalf("%q: reference returned %d tokens, %d probs", prompt, len(ref.Tokens), len(ref.Probs))
		}
		ref.Tokens = ref.Tokens[:oracleTokens]
		record = append(record, oracleRecord{Prompt: prompt, PromptIDs: ids, Tokens: ref.Tokens, Steps: stepTops(ref)})
		for _, exact := range []bool{true, false} {
			m.ExactGGML = exact
			got, l1 := generate(m, ids, oracleTokens)
			same := 0
			for same < oracleTokens && got[same] == ref.Tokens[same] {
				same++
			}
			lp := logSoftmax(l1)
			var worst float64
			// Compare log-probability differences to the reference's best
			// candidate: that is a logit difference, independent of the
			// normaliser (the server may renormalise over its candidates).
			top := ref.Probs[0].Top
			for _, c := range top {
				t.Logf("  id %d ref %.4f ours %.4f", c.ID, c.Logprob, lp[c.ID])
			}
			for _, c := range top[1:] {
				worst = math.Max(worst, math.Abs((lp[c.ID]-lp[top[0].ID])-(c.Logprob-top[0].Logprob)))
			}
			tol := f32LogprobTol
			if exact {
				tol = exactLogprobTol
			}
			t.Logf("%q exact=%v: %d/%d tokens equal, step-1 max |dlogprob| over top-10 = %.2e (tolerance %.0e)", prompt, exact, same, oracleTokens, worst, tol)
			if same < oracleTokens {
				gap := math.Inf(1)
				for _, c := range ref.Probs[same].Top {
					if c.ID == got[same] {
						gap = ref.Probs[same].Top[0].Logprob - c.Logprob
					}
				}
				t.Logf("  token %d differs: ours %d (%q), llama.cpp %d (%q); llama.cpp's own gap between them %.3f nats", same, got[same], tok.Piece(got[same]), ref.Tokens[same], tok.Piece(ref.Tokens[same]), gap)
				// A different token is accepted only at a near-tie of
				// llama.cpp's own distribution (numerical noise of the
				// summation order and of the activation quantisation).
				if gap > nearTie || same < minEqual {
					t.Errorf("%q exact=%v: token %d differs at gap %.3f (near-tie limit %.2f, at least %d equal tokens required)", prompt, exact, same, gap, nearTie, minEqual)
				}
			}
			if worst > tol {
				t.Errorf("%q exact=%v: step-1 log-probabilities differ by %.3e, tolerance %.0e", prompt, exact, worst, tol)
			}
		}
	}
}

func stepTops(r refResult) [][]refTop {
	var out [][]refTop
	for i := 0; i < oracleTokens && i < len(r.Probs); i++ {
		out = append(out, r.Probs[i].Top)
	}
	return out
}

const dequantFile = "SmolLM2-135M-Instruct-Q4_0-dequant-F32.gguf"

// TestOracleDequantizedF32 compares the f32-activation path with llama.cpp
// running the SAME weights dequantised to F32 by llama.cpp's own tool
// (llama-quantize --allow-requantize IN OUT F32): both sides then multiply f32
// activations by identical f32 weights, so what differs is only the order of
// summation. Skips without LLM_MODELS/dequantFile.
func TestOracleDequantizedF32(t *testing.T) {
	oracleDequantized(t, modelFile, dequantFile, "LLM_ORACLE_OUT_F32", true)
}

const (
	qwenFile        = "qwen2.5-0.5b-instruct-q4_0.gguf"
	qwenDequantFile = "qwen2.5-0.5b-instruct-q4_0-dequant-F32.gguf"
)

// TestOracleQwenDequantizedF32 is the same strict oracle for Qwen2.5-0.5B
// (NEOX rope, QKV biases, Q4_0 token embedding, separate Q8_0 output matrix,
// qwen2 pre-tokenizer): LLM_MODELS holds both files; the token ids of the
// prompts must equal llama.cpp's /tokenize, the f32 path its 32 greedy tokens.
func TestOracleQwenDequantizedF32(t *testing.T) {
	oracleDequantized(t, qwenFile, qwenDequantFile, "LLM_ORACLE_OUT_QWEN", false)
}

func oracleDequantized(t *testing.T, modelFile, dequantFile, outEnv string, alsoExact bool) {
	dir := os.Getenv("LLM_MODELS")
	if dir == "" {
		t.Skip("LLM_MODELS not set")
	}
	ref := filepath.Join(dir, dequantFile)
	if _, err := os.Stat(ref); err != nil {
		t.Skip(err)
	}
	fh, err := os.Open(filepath.Join(dir, modelFile))
	if err != nil {
		t.Skip(err)
	}
	defer fh.Close()
	st, _ := fh.Stat()
	f, err := gguf.Parse(fh, uint64(st.Size()))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewTokenizer(f)
	if err != nil {
		t.Fatal(err)
	}
	// f32 KV cache and no flash attention as well: llama.cpp's default cache is
	// f16, which alone moves the step-1 logits by about 2e-3 to 5e-3 nats.
	srv := startServer(t, ref, "--cache-type-k", "f32", "--cache-type-v", "f32", "-fa", "off")
	m, err := LoadCPU(f, fh, 256)
	if err != nil {
		t.Fatal(err)
	}
	var record []oracleRecord
	// Pre-tokenizer corner cases: contractions in both cases, newlines and
	// runs of spaces, digits, symbols, accents, a trailing space.
	for _, s := range []string{"It's DON'T I'LL we've  two  spaces", "a\n\n  b\r\n c\t d", "x = 3.14159 + 42;  # ok", "naïve café — 你好 😀", "end ", " \n", "'tis 'S 'Re"} {
		var tk struct {
			Tokens []int32 `json:"tokens"`
		}
		srv.post(t, "/tokenize", map[string]any{"content": s}, &tk)
		if ids := tok.Encode(s); !slices.Equal(ids, tk.Tokens) {
			t.Errorf("%q: tokenizer\n ours %v\n ref  %v", s, ids, tk.Tokens)
		}
	}
	for _, prompt := range oraclePrompts {
		var r refResult
		srv.post(t, "/completion", map[string]any{"prompt": prompt, "n_predict": oracleTokens, "temperature": 0,
			"return_tokens": true, "n_probs": 10, "cache_prompt": false, "seed": 1}, &r)
		var tk struct {
			Tokens []int32 `json:"tokens"`
		}
		srv.post(t, "/tokenize", map[string]any{"content": prompt}, &tk)
		ids := tok.Encode(prompt)
		if !slices.Equal(ids, tk.Tokens) {
			t.Errorf("%q: tokenizer\n ours %v\n ref  %v", prompt, ids, tk.Tokens)
			continue
		}
		r.Tokens = r.Tokens[:oracleTokens]
		record = append(record, oracleRecord{Prompt: prompt, PromptIDs: ids, Tokens: r.Tokens, Steps: stepTops(r)})
		modes := []bool{false}
		if alsoExact {
			modes = []bool{false, true}
		}
		for _, exact := range modes {
			m.ExactGGML = exact
			got, l1 := generate(m, ids, oracleTokens)
			same := 0
			for same < oracleTokens && got[same] == r.Tokens[same] {
				same++
			}
			lp := logSoftmax(l1)
			var worst float64
			top := r.Probs[0].Top
			for _, c := range top[1:] {
				worst = math.Max(worst, math.Abs((lp[c.ID]-lp[top[0].ID])-(c.Logprob-top[0].Logprob)))
			}
			t.Logf("%q f32 reference, ours exact=%v: %d/%d tokens equal, step-1 max |dlogprob| %.2e", prompt, exact, same, oracleTokens, worst)
			if !exact && (same != oracleTokens || worst > f32RefTol) {
				t.Errorf("%q: f32 path against the f32 reference: %d/%d tokens, step-1 difference %.2e (tolerance %.0e)", prompt, same, oracleTokens, worst, f32RefTol)
			}
		}
	}
	if out := os.Getenv(outEnv); out != "" && !t.Failed() {
		b, _ := json.MarshalIndent(record, "", " ")
		if err := os.WriteFile(out, append(b, '\n'), 0o644); err != nil {
			t.Error(err)
		}
	}
}

// f32RefTol: step-1 log-probability differences between the f32 path and
// llama.cpp on identical f32 weights (summation order only).
const f32RefTol = 1e-4 // measured 1.4e-5 at most

// TestQuantisedActivationsAmplifyRounding: with Q8_0 activations, a rounding
// difference of 1e-7 in the RMSNorm sum (float32 against float64) moves the
// step-1 logits by about as much as the whole quantisation noise: no
// emulation that is not bit-exact can match llama.cpp's Q4_0 run to 1e-3, and
// the f32 path does not have this property (the perturbation moves it by
// about 1e-6). Needs LLM_MODELS.
func TestQuantisedActivationsAmplifyRounding(t *testing.T) {
	dir := os.Getenv("LLM_MODELS")
	if dir == "" {
		t.Skip("LLM_MODELS not set")
	}
	fh, err := os.Open(filepath.Join(dir, modelFile))
	if err != nil {
		t.Skip(err)
	}
	defer fh.Close()
	st, _ := fh.Stat()
	f, _ := gguf.Parse(fh, uint64(st.Size()))
	tok, _ := NewTokenizer(f)
	m, _ := LoadCPU(f, fh, 256)
	defer func() { rmsF64 = false }()
	for _, prompt := range oraclePrompts {
		ids := tok.Encode(prompt)
		for _, exact := range []bool{false, true} {
			m.ExactGGML = exact
			rmsF64 = false
			_, a := generate(m, ids, 1)
			rmsF64 = true
			_, b := generate(m, ids, 1)
			la, lb := logSoftmax(a), logSoftmax(b)
			ta := int(Argmax(a))
			var worst float64
			for i := range la {
				if la[i] > la[ta]-6 { // the candidates that matter
					worst = math.Max(worst, math.Abs((la[i]-la[ta])-(lb[i]-lb[ta])))
				}
			}
			t.Logf("%q Q8 activations=%v: RMSNorm float32 against float64 moves the step-1 logits by %.2e nats", prompt, exact, worst)
			if !exact && worst > 1e-4 {
				t.Errorf("the f32 path moved by %.2e", worst)
			}
		}
	}
}
