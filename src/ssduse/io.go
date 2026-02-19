package ssduse

import (
	"bufio"
	"compress/zlib"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/lifejade/mm/src/matmult"
	"github.com/tuneinsight/lattigo/v5/ring"
)

const iopath string = "temp"

var basepath string = "none"
var Writers map[string]*PolyWriter

// init basepath, it must be first called
func workspaceDir() string {
	if basepath != "none" {
		return basepath
	}

	_, filename, _, _ := runtime.Caller(0)
	basepath = filepath.Dir(filename)
	basepath = filepath.Join(basepath, iopath)
	return basepath
}

type PolyWriter struct {
	name     string
	f        *os.File
	w        *bufio.Writer
	r        *bufio.Reader
	bufSize  int
	coeffBuf []byte
	offsets  []int64

	zw *zlib.Writer
	zr io.ReadCloser
}

func newPolyWriter(name string, bufSize int, N int) (*PolyWriter, error) {
	path := filepath.Join(workspaceDir(), name+".bin")

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	os.Remove(path)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}

	return &PolyWriter{
		name:     name,
		f:        f,
		w:        bufio.NewWriterSize(f, bufSize),
		r:        bufio.NewReaderSize(f, bufSize),
		coeffBuf: make([]byte, N*8),
		bufSize:  bufSize,
	}, nil
}

func (p *PolyWriter) closePolyWriter() error {
	if err := p.w.Flush(); err != nil {
		_ = p.f.Close()
		return err
	}
	return p.f.Close()
}

func (p *PolyWriter) appendPoly(val ring.Poly, level int) error {
	level = level + 1
	coeffBuf := p.coeffBuf

	for l := range level {
		if err := p.w.Flush(); err != nil {
			return err
		}

		pos, _ := p.f.Seek(0, io.SeekCurrent)
		p.offsets = append(p.offsets, pos)

		for i, v := range val.Coeffs[l] {
			binary.LittleEndian.PutUint64(coeffBuf[i*8:i*8+8], v)
		}

		if p.zw == nil {
			p.zw = zlib.NewWriter(p.w)
		} else {
			p.zw.Reset(p.w)
		}

		if _, err := p.zw.Write(coeffBuf); err != nil {
			return err
		}
		p.zw.Close()
	}
	return nil
}

func (p *PolyWriter) getPoly(idx int, val ring.Poly, level int) error {
	coeffBuf := p.coeffBuf
	numLevels := level + 1
	baseIdx := idx * numLevels

	for l := 0; l < numLevels; l++ {
		offset := p.offsets[baseIdx+l]
		if _, err := p.f.Seek(offset, io.SeekStart); err != nil {
			return err
		}

		zr, err := zlib.NewReader(p.f)
		if err != nil {
			return err
		}

		if _, err := io.ReadFull(zr, coeffBuf); err != nil {
			zr.Close()
			return err
		}
		zr.Close()

		for n := range val.Coeffs[l] {
			val.Coeffs[l][n] = binary.LittleEndian.Uint64(coeffBuf[n*8 : n*8+8])
		}
	}
	return nil
}

func (p *PolyWriter) appendPoly32(val matmult.Poly, level int) error {
	level = level + 1
	coeffBuf := p.coeffBuf[:len(p.coeffBuf)/2]

	for l := range level {
		if err := p.w.Flush(); err != nil {
			return err
		}

		pos, _ := p.f.Seek(0, io.SeekCurrent)
		p.offsets = append(p.offsets, pos)

		for i, v := range val.Coeffs[l] {
			binary.LittleEndian.PutUint32(coeffBuf[i*4:i*4+4], v)
		}

		if p.zw == nil {
			p.zw = zlib.NewWriter(p.w)
		} else {
			p.zw.Reset(p.w)
		}

		if _, err := p.zw.Write(coeffBuf); err != nil {
			return err
		}
		p.zw.Close()
	}
	return nil
}

// getPoly: 인덱스를 기반으로 특정 다항식을 복원합니다.
func (p *PolyWriter) getPoly32(idx int, val matmult.Poly, level int) error {
	coeffBuf := p.coeffBuf[:len(p.coeffBuf)/2]
	numLevels := level + 1
	baseIdx := idx * numLevels

	for l := 0; l < numLevels; l++ {
		// 저장된 오프셋으로 이동
		offset := p.offsets[baseIdx+l]
		if _, err := p.f.Seek(offset, io.SeekStart); err != nil {
			return err
		}

		// zlib Reader 생성 및 압축 해제
		zr, err := zlib.NewReader(p.f)
		if err != nil {
			return err
		}

		if _, err := io.ReadFull(zr, coeffBuf); err != nil {
			zr.Close()
			return err
		}
		zr.Close()

		// 역직렬화
		for n := range val.Coeffs[l] {
			val.Coeffs[l][n] = binary.LittleEndian.Uint32(coeffBuf[n*4 : n*4+4])
		}
	}
	return nil
}

// func (p *PolyWriter) getPoly32OnlyPi(idx int, val matmult.Poly, level, levelidx int) error {
// 	coeffBuf := p.coeffBuf[:len(p.coeffBuf)/2]
// 	numLevels := level + 1
// 	baseIdx := idx * numLevels

// 	for l := 0; l < numLevels; l++ {
// 		// 저장된 오프셋으로 이동
// 		offset := p.offsets[baseIdx+l]
// 		if _, err := p.f.Seek(offset, io.SeekStart); err != nil {
// 			return err
// 		}

// 		// zlib Reader 생성 및 압축 해제
// 		zr, err := zlib.NewReader(p.f)
// 		if err != nil {
// 			return err
// 		}

// 		if _, err := io.ReadFull(zr, coeffBuf); err != nil {
// 			zr.Close()
// 			return err
// 		}
// 		zr.Close()

// 		// 역직렬화
// 		for n := range val.Coeffs[l] {
// 			val.Coeffs[l][n] = binary.LittleEndian.Uint32(coeffBuf[n*4 : n*4+4])
// 		}
// 	}
// 	return nil
// }

func (p *PolyWriter) flush() error {
	return p.w.Flush()
}

func AddWriters(name string, bufSize, N int) {
	if Writers == nil {
		Writers = make(map[string]*PolyWriter)
	}

	if Writers[name] == nil {
		Writers[name], _ = newPolyWriter(name, bufSize, N)
	}
}

func CloseAllWriters() {
	for _, v := range Writers {
		v.closePolyWriter()
	}
}
func FlushPoly(name string) {
	if Writers[name] == nil {
		return
	}
	Writers[name].flush()
}
func AppendPoly(name string, val ring.Poly, level int) {
	if Writers[name] == nil {
		return
	}
	Writers[name].appendPoly(val, level)
}

func GetPoly(name string, idx int, val ring.Poly, level int) {
	if Writers[name] == nil {
		return
	}
	_ = Writers[name].getPoly(idx, val, level)
}

func AppendPoly32(name string, val matmult.Poly, level int) {
	if Writers[name] == nil {
		return
	}
	Writers[name].appendPoly32(val, level)
}

func GetPoly32(name string, idx int, val matmult.Poly, level int) {
	if Writers[name] == nil {
		return
	}
	_ = Writers[name].getPoly32(idx, val, level)
}
