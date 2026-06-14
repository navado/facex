# i.MX 8M Plus NPU — reproduction runbook

End-to-end steps to reproduce FaceX/MobileNet on the VIP9000 NPU of a **CompuLab IOT-GATE-IMX8PLUS**
running plain **Debian 12** (not an NXP Yocto image). Validated 2026-06-14.

## 0. The version-match rule (read first)

The board's Vivante `galcore` driver is **built into CompuLab's kernel** (`/dev/galcore`, no loadable
`.ko`). The userspace must ABI-match it exactly or you get "0 ops delegated" / hang / shader-compile errors.

```sh
# on board (root): the ABI contract
sudo mount -t debugfs none /sys/kernel/debug 2>/dev/null
cat /sys/kernel/debug/gc/version          # -> 6.4.11.p2.745085  (this board)
```

That maps to NXP **LF6.6.3_1.0.0** (Yocto nanbield) → `imx-gpu-viv 6.4.11.p2.4` + TFLite 2.14 + tim-vx +
vx_delegate. Use exactly that release.

## 1. Get the matching userspace (two parts)

**a) Vivante/OVX driver** — open NXP mirror, no login:
```sh
curl -O https://www.nxp.com/lgfiles/NMG/MAD/YOCTO/imx-gpu-viv-6.4.11.p2.4-aarch64-b07999b.bin
# sha256 ba86656c357c5d9793058695f320e4cf650d4693e84321870bad392f2a622807
sh imx-gpu-viv-6.4.11.p2.4-aarch64-b07999b.bin --auto-accept --force   # -> gpu-core/usr/{lib,include}
```

**b) TFLite + tim-vx + vx_delegate** — from the LF6.6.3 rootfs (NXP account/EULA):
download `LF_v6.6.3-1.0.0_images_IMX8MPEVK.zip`, then pull only the rootfs tarball and extract the libs
(no need to unpack the 10 GB `.wic`):
```sh
unzip -p LF_v6.6.3-1.0.0_images_IMX8MPEVK.zip imx-image-full-imx8mpevk.tar.zst > rootfs.tar.zst
zstd -dc rootfs.tar.zst | tar -x \
  './usr/lib/libtensorflow-lite.so*' './usr/lib/libvx_delegate.so*' './usr/lib/libtim-vx.so*' \
  './usr/lib/libGAL.so*' './usr/lib/libVSC.so*' './usr/lib/libCLC.so*' './usr/lib/libGLSLC.so*' \
  './usr/lib/libOpenVX*' './usr/lib/libOpenCL.so*' './usr/lib/libArchModelSw.so*' \
  './usr/lib/libNNArchPerf.so*' './usr/lib/libNN*' './usr/lib/libOvx*' \
  './usr/bin/tensorflow-lite-2.14.0/examples/benchmark_model' \
  './usr/bin/tensorflow-lite-2.14.0/examples/mobilenet_v1_1.0_224_quant.tflite'
```
Stage everything under `~/npu/eiq/usr/` on the board (`lib/`, `bin/`, and the gpu-viv `include/` from
step a — the shader compiler needs `include/CL/cl_viv_vx_ext.h`). Create the TFLite sonames:
```sh
cd ~/npu/eiq/usr/lib && ln -sf libtensorflow-lite.so.2.14.0 libtensorflow-lite.so.2
```

## 2. Runtime container

The eIQ libs are Yocto-built (glibc ≥ 2.38) and won't run on Debian bookworm's glibc 2.36 — run them in a
newer-glibc container. The board's Docker can't pull (Tailscale DNS), so **build on another host and load**:
```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      libstdc++6 libgomp1 libdrm2 libwayland-client0 libwayland-server0 \
      libwayland-egl1 libegl1 libgles2 libgbm1 && rm -rf /var/lib/apt/lists/*
```
```sh
docker build --platform linux/arm64 -t facex-npu-rt:2404 .
docker save facex-npu-rt:2404 | gzip | ssh compulab@BOARD 'gunzip | docker load'
```

## 3. Run on the NPU

```sh
docker run --rm \
  --device /dev/galcore --device /dev/dri/renderD128 --device /dev/dri/card0 \
  -v ~/npu/eiq/usr:/eiq:ro -v ~/npu/models:/models:ro \
  -e LD_LIBRARY_PATH=/eiq/lib -e VIVANTE_SDK_DIR=/eiq -e USE_GPU_INFERENCE=0 \
  facex-npu-rt:2404 \
  /eiq/bin/tensorflow-lite-2.14.0/examples/benchmark_model \
    --graph=/models/edgeface_xs_int8.tflite --num_threads=1 \
    --num_runs=30 --warmup_runs=3 \
    --external_delegate_path=/eiq/lib/libvx_delegate.so
```
Expect: `Explicitly applied EXTERNAL delegate, and the model graph will be completely executed by the
delegate`, a multi-second first inference (one-time NPU graph compile), then ~25.8 ms steady-state for
EdgeFace-XS INT8 (2.93 ms for `mobilenet_v1_1.0_224_quant.tflite`).

Validate the NPU lights up with the stock MobileNet first — it must show non-zero delegation before
trying FaceX.

## 4. (Re)generate the EdgeFace TFLite models

Both run on an x86 or Apple-Silicon host with Docker (not the board). Scripts in `../../tools/imx8mp/`.

1. **Export ONNX** (PyTorch → ONNX, GELU tanh-approx so no Erf/Flex op):
   `tools/imx8mp/export_edgeface_onnx.sh` (runs in a `python:3.11-slim` container; downloads the upstream
   `otroshi/edgeface` `edgeface_xs_gamma_06` weights).
2. **Convert + quantize** with TensorFlow **2.14** (matches the board's TFLite 2.14 op versions —
   newer TF emits e.g. `SQRT` v2 which the 2.14 runtime rejects):
   `tools/imx8mp/convert_tflite.sh` (onnx2tf → saved_model → FP32 + INT8 TFLite, with a representative
   dataset of aligned face crops in `calib/`).

> **Op/version gotchas learned the hard way:** (1) keep GELU as tanh-approx or you get a `FlexErf` op that
> the NPU can't run; (2) convert with TF 2.14 to match the runtime's op versions; (3) keep model I/O
> **float32** (int8 output crushes the small-magnitude embedding); (4) onnx2tf on arm64: pin
> `onnxsim==0.6.5` (wheel, drops the no-arm64-wheel `onnxoptimizer`), use a Debian **bookworm** base
> (trixie breaks h5py wheel selection), and it needs `tf_keras` + `onnx_graphsurgeon` + `sng4onnx` +
> `psutil` + `ai_edge_litert` at runtime even though they aren't declared deps.

## 5. Known limitations

- **INT8 EdgeFace accuracy is not production-usable** (cosine ~0.29 vs FP32). Needs QAT + a real
  aligned-face calibration set. See `README.md`.
- **FP32 does not run on the VIP9000** (F32 EVIS GEMM shader fails to verify). INT8 is the NPU path.
- **Detector stays on CPU** (`facex_npu_detect` is `-ENOSYS`); recommended deployment is hybrid
  (CPU detect via `libfacex.a` + NPU embed).
- EdgeFace weights are **CC BY-NC-SA 4.0** — evaluation only, not for commercial artifacts.
