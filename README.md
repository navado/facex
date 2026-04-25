# FaceX

Fast face embedding library. 3ms inference, 7MB binary, zero dependencies.

Handwritten C + AVX2/AVX-512. Faster than ONNX Runtime on CPU.

## Benchmarks

EdgeFace-XS (1.77M params, 99.73% LFW) on Intel i5-11500:

| Engine | Median | Min |
|--------|--------|-----|
| **FaceX** | **3.0 ms** | **2.87 ms** |
| ONNX Runtime 1.23 | 3.9 ms | 3.18 ms |
| InsightFace (R34) | 17 ms | — |
| dlib | 50-80 ms | — |

## Quick Start

```c
#include "facex.h"

FaceX* fx = facex_init("edgeface_xs_fp32.bin", NULL);

float embedding[512];
facex_embed(fx, rgb_112x112_hwc, embedding);

float sim = facex_similarity(emb1, emb2);
// sim > 0.3 → same person

facex_free(fx);
```

## Build

```bash
make        # builds libfacex.a + facex-cli
make example
./facex-example edgeface_xs_fp32.bin
```

Requirements: GCC with AVX2 support. No other dependencies.

## API

```c
FaceX* facex_init(const char* weights_path, const char* license_key);
int    facex_embed(FaceX* fx, const float* rgb_hwc, float embedding[512]);
float  facex_similarity(const float emb1[512], const float emb2[512]);
void   facex_free(FaceX* fx);
```

**Input:** 112x112 RGB image as float32 array in HWC layout, values normalized to [-1, 1].

**Output:** 512-dimensional L2-normalized embedding vector.

## Go Binding

```go
import "github.com/facex-engine/facex/go/facex"

ff, _ := facex.New(facex.Config{
    Exe:     "./facex-cli",
    Weights: "./edgeface_xs_fp32.bin",
})
embedding, _ := ff.Embed(rgbImage)
```

## Architecture

- Pure C99 with SIMD intrinsics (AVX2, FMA, AVX-512 VNNI)
- INT8 quantized GEMM with per-channel scaling
- FP32 packed column-panel MatMul
- Thread pool for parallel computation
- Optional AES-256 weight encryption with hardware binding

## Model

Uses [EdgeFace-XS](https://github.com/ahmadnassri/edgeface) (CC BY-NC-SA 4.0):
- 1.77M parameters
- 99.73% accuracy on LFW
- 112x112 input, 512-dim output

## License

CC BY-NC-SA 4.0 (follows upstream model license).

For commercial licensing, contact: [bauratynov@gmail.com](mailto:bauratynov@gmail.com)

## Author

Created by **Baurzhan Atinov** — [GitHub](https://github.com/bauratynov)
