# FaceX on the i.MX 8M Plus NPU (VIP9000 / VxDelegate)

Hardware-validated results, converted models, and reproduction steps for running FaceX
on the **CompuLab IOT-GATE-IMX8PLUS** (i.MX 8M Plus, quad Cortex-A53, Debian 12).

Full step-by-step: [`runbook.md`](runbook.md).

## Results (validated 2026-06-14)

Board: i.MX8MP, galcore **6.4.11.p2.745085** (builtin). Stack: NXP eIQ from LF6.6.3
(`imx-gpu-viv 6.4.11.p2.4`, TFLite 2.14, tim-vx, vx_delegate) run in a container against `/dev/galcore`.

Latency = mean inference time; throughput = single-stream inferences/sec (`1000/mean_ms`).

| Model | CPU latency | CPU thr | **NPU latency** | **NPU thr** | Speedup | Notes |
|---|--:|--:|--:|--:|--:|---|
| MobileNetV1 1.0 224 INT8 | 42.2 ms | 23.7 inf/s | **2.93 ms** | **341 inf/s** | **14.4×** | NPU bring-up validation; full delegation |
| EdgeFace-XS INT8 (this repo) | 145.3 ms | 6.9 inf/s | **25.8 ms** | **38.8 inf/s** | 5.6× | Full delegation; first-run 41 s = one-time graph compile |
| EdgeFace-XS FP32 | — | — | (fails) | — | — | VIP9000 F32 EVIS GEMM shader does not verify; INT8 is the NPU path |

(CPU = 4× A53 via TFLite/XNNPACK. NPU = VIP9000 via VxDelegate. Throughput is single-stream — one
inference at a time; the NPU already runs the whole graph, so concurrent-stream throughput is ≈ the same.)

For reference, the hand-tuned FaceX NEON CPU engine embeds in ~58.2 ms / **17.2 emb/s** (see
`../bench/imx8mp_baseline.csv`), so the NPU INT8 path is ~2.3× the throughput of the best CPU path —
**for latency/throughput; accuracy caveat below.**

## ⚠️ Accuracy caveat (important)

- **FP32 TFLite is numerically exact**: `cosine(tflite_fp32, torch reference) = 1.000000`.
- **INT8 TFLite accuracy is NOT production-usable**: `cosine(int8, fp32) ≈ 0.29`. EdgeFace-XS's
  LayerNorm + XCA cross-covariance attention quantize poorly under **post-training quantization**,
  even with the calibration sample in the calibration set — this is architectural, not calibration
  coverage. Producing a usable INT8 EdgeFace needs **quantization-aware training (QAT)** and/or a
  proper aligned-face calibration dataset (~100+ ArcFace-aligned 112×112 crops). Neither was available
  here. The INT8 model below is a **feasibility/latency artifact**, not a working recognizer.

So: the NPU itself is fully validated (MobileNet 14×, exact); EdgeFace *runs* on the NPU and we have
its latency, but INT8 *accuracy* for EdgeFace specifically is an open QAT item.

## Files

| File | What it is |
|---|---|
| `edgeface_xs_fp32.tflite` | EdgeFace-XS, FP32, TFLite 2.14 ops. **Numerically exact** (cosine 1.0 vs upstream torch). Runs on CPU/XNNPACK; FP32 does not run on this NPU. |
| `edgeface_xs_int8.tflite` | EdgeFace-XS, full-INT8, TFLite 2.14 ops. **Runs fully on the VIP9000 NPU at 25.8 ms.** Accuracy WIP (cosine ~0.29 — needs QAT). |

## Provenance & license

- Architecture + weights: **EdgeFace-XS (`edgeface_xs_gamma_06`)** from the upstream EdgeFace project
  (`github.com/otroshi/edgeface`, Idiap), the same source FaceX's `data/edgeface_xs_fp32.bin` derives from.
- Conversion: PyTorch → ONNX (opset 13, GELU tanh-approx to keep ops NPU-friendly) → TFLite via `onnx2tf`,
  re-quantized with TensorFlow 2.14 to match the board runtime. Scripts: `../../tools/imx8mp/`.
- **License: the EdgeFace weights are CC BY-NC-SA 4.0 (non-commercial).** These `.tflite` files inherit that
  license and are included for evaluation/reproduction only — **do not ship them in commercial artifacts.**
  FaceX engine code remains Apache-2.0.
