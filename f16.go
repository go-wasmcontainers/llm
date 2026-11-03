package llm

import "math"

// F16ToF32 converts an IEEE binary16 value, exactly.
func F16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch {
	case exp == 0:
		if man == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal: normalise
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		man &= 0x3ff
		return math.Float32frombits(sign | e<<23 | man<<13)
	case exp == 31:
		return math.Float32frombits(sign | 0xff<<23 | man<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | man<<13)
}

// F32ToF16 converts to binary16, round to nearest even.
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xff
	man := b & 0x7fffff
	if exp == 0xff {
		if man != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	e := exp - 127 + 15
	if e >= 31 {
		return sign | 0x7c00
	}
	if e <= 0 {
		if e < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint32(14 - e)
		h := man >> shift
		rem := man & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || rem == half && h&1 == 1 {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(e)<<10 | man>>13
	rem := man & 0x1fff
	if rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++
	}
	return sign | uint16(h)
}

// RoundF16 rounds f through binary16 (what an f16 KV cache stores).
func RoundF16(f float32) float32 { return F16ToF32(F32ToF16(f)) }
