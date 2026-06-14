# FaceX cumulative benchmark — cross-platform

One table across every platform measured so far. `embed` = EdgeFace-XS (512-d) only;
`e2e` = detect + align + embed. **throughput = single-stream inferences/sec** (`1000 / latency`);
the engine already uses all cores per inference, so this is the sustained 1-stream rate.

Sources: `m2_baseline.csv` (Apple M2), `imx8mp_baseline.csv` (i.MX8MP CPU),
`../imx8mp_npu/README.md` (i.MX8MP NPU). Regenerate per-host with `scripts/bench_all.sh`.

## Embed (EdgeFace-XS, 512-d)

| Platform | Backend | latency (ms) | **throughput (inf/s)** | Notes |
|---|---|--:|--:|---|
| Apple M2 | Accelerate / AMX | 3.58 (med) | **280** | `make ACCELERATE=1`; fastest host path |
| Apple M2 | NEON (default) | 4.05 (med) | **247** | portable default |
| **i.MX8MP** | **VIP9000 NPU / VxDelegate INT8** | **25.8 (mean)** | **38.8** | full delegation; ⚠ INT8 PTQ accuracy WIP (cosine 0.29 — needs QAT) |
| i.MX8MP | 4× A53 NEON (FP32, row-parallel MLP) | 58.2 (med) | 17.2 | hand-tuned engine; numerically exact |
| i.MX8MP | 4× A53 NEON (FP32, single-core) | 69.8 (med) | 14.3 | before MLP threading |
| i.MX8MP | 4× A53 TFLite INT8 (CPU) | 145.3 (mean) | 6.9 | TFLite ref interpreter, not the FaceX engine |

## End-to-end (detect + align + embed)

| Platform | Backend | latency (ms) | **throughput (fps)** | Notes |
|---|---|--:|--:|---|
| Apple M2 | Accelerate / AMX | 7.75 (med) | **129** | 1 face in synthetic frame |
| Apple M2 | NEON (default) | 7.89 (med) | 127 | 1 face |
| i.MX8MP | 4× A53 NEON | 59.9 (med) | 16.7 | detector cost (synthetic frame, no face) |

## Reference: MobileNetV1 1.0 224 INT8 (NPU bring-up validation)

| Platform | Backend | latency (ms) | **throughput (inf/s)** |
|---|---|--:|--:|
| i.MX8MP | VIP9000 NPU / VxDelegate | 2.93 (mean) | **341** |
| i.MX8MP | 4× A53 CPU (TFLite) | 42.2 (mean) | 23.7 |

## Takeaways

- **Apple M2 (AMX) is the throughput leader** for EdgeFace embed (~280 inf/s) — it's a laptop-class part.
- On the **i.MX8MP edge SoC**, the **NPU gives the best latency/throughput** (38.8 inf/s embed,
  ~2.3× the best CPU path) — but EdgeFace's INT8 *accuracy* via post-training quantization is not yet
  usable; that's the open QAT item. MobileNet (341 inf/s, 14.4× over CPU) shows the NPU's real ceiling
  on a quantization-friendly model.
- On A53 CPU, EdgeFace-XS is **memory-bandwidth bound**: 4-core threading buys only ~1.2× over 1 core.
