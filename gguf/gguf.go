// Package gguf parses the header of a GGUF file (ggml docs/gguf.md, version
// 3): metadata key/values and tensor infos. It reads through an io.ReaderAt
// and never loads tensor data, so a model larger than memory can be opened.
//
// Every count and length read from the file is bounded before it is used to
// allocate (the file is untrusted), and offsets are checked in uint64.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Bounds on a header. A violation is ErrBound.
const (
	MaxTensors   = 1 << 16
	MaxKV        = 1 << 16
	MaxString    = 1 << 24
	MaxArray     = 1 << 22
	MaxHeader    = 256 << 20
	MaxDims      = 4
	MaxAlignment = 1 << 20
)

// Errors.
var (
	ErrMagic   = errors.New("gguf: bad magic")
	ErrVersion = errors.New("gguf: unsupported version")
	ErrBound   = errors.New("gguf: a size or count exceeds its bound")
	ErrFormat  = errors.New("gguf: malformed header")
)

// ValueType is a metadata value type.
type ValueType uint32

// Metadata value types (gguf.md, gguf_metadata_value_type).
const (
	U8 ValueType = iota
	I8
	U16
	I16
	U32
	I32
	F32
	Bool
	String
	Array
	U64
	I64
	F64
)

// Type is a ggml tensor type (ggml_type).
type Type uint32

// Tensor types used by the models of this repository.
const (
	TypeF32  Type = 0
	TypeF16  Type = 1
	TypeQ4_0 Type = 2
	TypeQ4_1 Type = 3
	TypeQ5_0 Type = 6
	TypeQ5_1 Type = 7
	TypeQ8_0 Type = 8
	TypeQ6_K Type = 14
)

// BlockSize and TypeSize give the elements per block and bytes per block.
func (t Type) BlockSize() (elems, bytes uint64, ok bool) {
	switch t {
	case TypeF32:
		return 1, 4, true
	case TypeF16:
		return 1, 2, true
	case TypeQ4_0:
		return 32, 18, true
	case TypeQ4_1:
		return 32, 20, true
	case TypeQ5_0:
		return 32, 22, true
	case TypeQ5_1:
		return 32, 24, true
	case TypeQ8_0:
		return 32, 34, true
	case TypeQ6_K:
		return 256, 210, true
	}
	return 0, 0, false
}

// Tensor is a tensor info. Offset is relative to the start of the data
// section; use File.DataOffset to get the absolute position.
type Tensor struct {
	Name   string
	Dims   []uint64 // ne[0] first (the contiguous dimension)
	Type   Type
	Offset uint64
	Size   uint64 // bytes
}

// Elems is the number of elements.
func (t *Tensor) Elems() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// File is a parsed header.
type File struct {
	Version    uint32
	Meta       map[string]any // string, bool, ints as int64/uint64, floats as float64, arrays as []any
	Tensors    []Tensor
	byName     map[string]int
	DataOffset uint64 // absolute offset of the data section
	Alignment  uint64
	Size       uint64 // file size, as given to Parse
}

// Tensor returns the tensor called name.
func (f *File) Tensor(name string) (*Tensor, bool) {
	i, ok := f.byName[name]
	if !ok {
		return nil, false
	}
	return &f.Tensors[i], true
}

type reader struct {
	r      io.ReaderAt
	win    []byte // read-ahead window
	winOff uint64
	pos    uint64
	lim    uint64
	buf    [8]byte
}

// readAhead is the size of the window the header is read through: a header
// is thousands of tiny fields, and in a container each ReadAt is a system
// call that crosses into the host.
const readAhead = 1 << 20

func (r *reader) read(p []byte) error {
	if r.pos+uint64(len(p)) > r.lim {
		return fmt.Errorf("%w: header runs past byte %d", ErrBound, r.lim)
	}
	if uint64(len(p)) >= readAhead {
		n, err := r.r.ReadAt(p, int64(r.pos))
		if n < len(p) {
			if err == nil || err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return err
		}
		r.pos += uint64(len(p))
		return nil
	}
	if r.pos < r.winOff || r.pos+uint64(len(p)) > r.winOff+uint64(len(r.win)) {
		if r.win == nil {
			r.win = make([]byte, readAhead)
		}
		n, err := r.r.ReadAt(r.win[:min(uint64(readAhead), r.lim-r.pos)], int64(r.pos))
		r.winOff, r.win = r.pos, r.win[:n]
		if n < len(p) {
			if err == nil || err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return err
		}
		r.win = r.win[:n:readAhead]
	}
	copy(p, r.win[r.pos-r.winOff:])
	r.pos += uint64(len(p))
	return nil
}

func (r *reader) u32() (uint32, error) {
	if err := r.read(r.buf[:4]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(r.buf[:4]), nil
}

func (r *reader) u64() (uint64, error) {
	if err := r.read(r.buf[:8]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(r.buf[:8]), nil
}

func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > MaxString {
		return "", fmt.Errorf("%w: string of %d bytes", ErrBound, n)
	}
	b := make([]byte, n)
	if err := r.read(b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *reader) value(t ValueType, depth int) (any, error) {
	switch t {
	case U8, I8, Bool:
		if err := r.read(r.buf[:1]); err != nil {
			return nil, err
		}
		switch t {
		case U8:
			return uint64(r.buf[0]), nil
		case I8:
			return int64(int8(r.buf[0])), nil
		}
		return r.buf[0] != 0, nil
	case U16, I16:
		if err := r.read(r.buf[:2]); err != nil {
			return nil, err
		}
		v := binary.LittleEndian.Uint16(r.buf[:2])
		if t == U16 {
			return uint64(v), nil
		}
		return int64(int16(v)), nil
	case U32, I32, F32:
		v, err := r.u32()
		if err != nil {
			return nil, err
		}
		switch t {
		case U32:
			return uint64(v), nil
		case I32:
			return int64(int32(v)), nil
		}
		return float64(math.Float32frombits(v)), nil
	case U64, I64, F64:
		v, err := r.u64()
		if err != nil {
			return nil, err
		}
		switch t {
		case U64:
			return v, nil
		case I64:
			return int64(v), nil
		}
		return math.Float64frombits(v), nil
	case String:
		return r.str()
	case Array:
		if depth > 0 {
			return nil, fmt.Errorf("%w: nested array", ErrFormat)
		}
		et, err := r.u32()
		if err != nil {
			return nil, err
		}
		n, err := r.u64()
		if err != nil {
			return nil, err
		}
		if n > MaxArray {
			return nil, fmt.Errorf("%w: array of %d", ErrBound, n)
		}
		// Each element takes at least one byte: refuse a count the
		// remaining header cannot hold, before allocating.
		if n > r.lim-r.pos {
			return nil, fmt.Errorf("%w: array of %d past the header", ErrBound, n)
		}
		out := make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			v, err := r.value(ValueType(et), depth+1)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("%w: value type %d", ErrFormat, t)
}

// Parse reads the header of a file of the given size.
func Parse(r io.ReaderAt, size uint64) (*File, error) {
	rd := &reader{r: r, lim: min(size, MaxHeader)}
	var magic [4]byte
	if err := rd.read(magic[:]); err != nil {
		return nil, err
	}
	if string(magic[:]) != "GGUF" {
		return nil, ErrMagic
	}
	ver, err := rd.u32()
	if err != nil {
		return nil, err
	}
	if ver != 3 {
		return nil, fmt.Errorf("%w: %d", ErrVersion, ver)
	}
	nt, err := rd.u64()
	if err != nil {
		return nil, err
	}
	nkv, err := rd.u64()
	if err != nil {
		return nil, err
	}
	if nt > MaxTensors || nkv > MaxKV {
		return nil, fmt.Errorf("%w: %d tensors, %d metadata entries", ErrBound, nt, nkv)
	}
	f := &File{Version: ver, Meta: make(map[string]any), byName: make(map[string]int), Size: size}
	for i := uint64(0); i < nkv; i++ {
		k, err := rd.str()
		if err != nil {
			return nil, err
		}
		t, err := rd.u32()
		if err != nil {
			return nil, err
		}
		v, err := rd.value(ValueType(t), 0)
		if err != nil {
			return nil, fmt.Errorf("key %q: %w", k, err)
		}
		f.Meta[k] = v
	}
	f.Alignment = 32
	if a, ok := f.Meta["general.alignment"]; ok {
		av, ok := a.(uint64)
		if !ok || av == 0 || av > MaxAlignment || av%8 != 0 {
			return nil, fmt.Errorf("%w: general.alignment %v", ErrFormat, a)
		}
		f.Alignment = av
	}
	f.Tensors = make([]Tensor, 0, min(nt, 1024))
	for i := uint64(0); i < nt; i++ {
		name, err := rd.str()
		if err != nil {
			return nil, err
		}
		nd, err := rd.u32()
		if err != nil {
			return nil, err
		}
		if nd == 0 || nd > MaxDims {
			return nil, fmt.Errorf("%w: tensor %q has %d dimensions", ErrFormat, name, nd)
		}
		t := Tensor{Name: name, Dims: make([]uint64, nd)}
		elems := uint64(1)
		for j := range t.Dims {
			d, err := rd.u64()
			if err != nil {
				return nil, err
			}
			if d == 0 || d > 1<<32 {
				return nil, fmt.Errorf("%w: tensor %q dimension %d", ErrBound, name, d)
			}
			if elems > (1<<62)/d {
				return nil, fmt.Errorf("%w: tensor %q too large", ErrBound, name)
			}
			elems *= d
			t.Dims[j] = d
		}
		ty, err := rd.u32()
		if err != nil {
			return nil, err
		}
		t.Type = Type(ty)
		off, err := rd.u64()
		if err != nil {
			return nil, err
		}
		t.Offset = off
		be, bb, ok := t.Type.BlockSize()
		if !ok {
			return nil, fmt.Errorf("%w: tensor %q has unsupported ggml type %d", ErrFormat, name, ty)
		}
		if t.Dims[0]%be != 0 {
			return nil, fmt.Errorf("%w: tensor %q row of %d is not a multiple of %d", ErrFormat, name, t.Dims[0], be)
		}
		t.Size = elems / be * bb
		if _, dup := f.byName[name]; dup {
			return nil, fmt.Errorf("%w: duplicate tensor %q", ErrFormat, name)
		}
		f.byName[name] = len(f.Tensors)
		f.Tensors = append(f.Tensors, t)
	}
	f.DataOffset = (rd.pos + f.Alignment - 1) / f.Alignment * f.Alignment
	for i := range f.Tensors {
		t := &f.Tensors[i]
		if t.Offset%f.Alignment != 0 {
			return nil, fmt.Errorf("%w: tensor %q offset %d is not aligned", ErrFormat, t.Name, t.Offset)
		}
		if t.Offset > size || f.DataOffset > size-t.Offset || t.Size > size-f.DataOffset-t.Offset {
			return nil, fmt.Errorf("%w: tensor %q lies past the end of the file", ErrBound, t.Name)
		}
	}
	return f, nil
}

// Uint returns an unsigned integer metadata value.
func (f *File) Uint(key string) (uint64, bool) {
	switch v := f.Meta[key].(type) {
	case uint64:
		return v, true
	case int64:
		if v >= 0 {
			return uint64(v), true
		}
	}
	return 0, false
}

// Float returns a float metadata value.
func (f *File) Float(key string) (float64, bool) {
	v, ok := f.Meta[key].(float64)
	return v, ok
}

// Str returns a string metadata value.
func (f *File) Str(key string) (string, bool) {
	v, ok := f.Meta[key].(string)
	return v, ok
}
