# i.MX 8M Plus NPU bring-up — Docker, CompuLab-GitHub-sourced userspace

Goal: run FaceX **embed on the VIP9000 NPU** (VxDelegate) on the CompuLab IOT-GATE-IMX8PLUS,
target ~8–15 ms vs the validated ~58.9 ms CPU baseline (`docs/bench/imx8mp_baseline.csv`).

> ## ✅ NPU VALIDATED ON HARDWARE (2026-06-13)
> The bring-up works. MobileNetV1 1.0 224 quant: **CPU(4×A53) 42.2 ms → NPU(VxDelegate) 2.93 ms (~14×)**,
> full graph delegation, in a container against `/dev/galcore`. The exact-matched p2.4 gpu-viv drives the
> board's builtin galcore p2.745085. **Working recipe + numbers are in the `imx8mp_baseline` memory.**
> Stack: libs from `LF_v6.6.3-1.0.0_images_IMX8MPEVK.zip` rootfs tarball (TFLite 2.14 + tim-vx + vx_delegate
> + gpu-viv) → `~/npu/eiq/usr/` on board, run in `facex-npu-rt:2404` (ubuntu:24.04 + Vivante deps, built on
> Mac & `docker load`ed because the board's Tailscale DNS can't pull/apt).
> **Remaining:** EdgeFace-XS → INT8 tflite → run via vx_delegate → verify vs CPU `.bin` → bench.

This plan supersedes the "Missing components" section of `imx8mp_plan.md` with a concrete,
container-based route. Decisions taken (2026-06-04): **source the userspace from CompuLab's
GitHub BSP / the NXP release it pins**, and **host it in a Docker container** (the board already
runs Docker; this keeps the fragile Vivante stack out of the Debian rootfs).

## The one rule that governs everything: version match

The board's **galcore driver is built into CompuLab's kernel** (`/dev/galcore` 199:0 present, no
loadable `galcore.ko`, DRI node `40000000.mix_gpu_ml`). Built-in ⇒ you cannot bump it without a
kernel rebuild. The Vivante userspace (`libGAL/libOpenVX/libVSC/libvx_delegate/libtim-vx` + TFLite)
**must ABI-match that galcore version**, or you get the classic failure modes:

- delegate `dlopen`s but **0 nodes delegated** (silent CPU fallback), or
- `Invoke` **hangs / times out** (NPU executes garbage), or
- `galcore` userspace version check fails at init.

So the entire sourcing strategy is: *derive the container userspace from the exact NXP release
CompuLab's kernel was built from.*

## The version chain (galcore CONFIRMED from the board, 2026-06-04)

| Layer | Value | Evidence |
|---|---|---|
| Board kernel | `6.6.3-gf0f789e68d79` (built Jul 2024) | `uname -r` on board |
| GPU/NPU driver | **galcore `6.4.11.p2.745085`** (builtin, `CONFIG_MXC_GPU_VIV=y`) | ✅ `cat /sys/kernel/debug/gc/version` on board |
| CompuLab BSP | `compulab-yokneam/meta-bsp-imx8mp` (their kernel = NXP LF6.6.3) | matches board kernel 6.6.3 |
| NXP manifest | **`nanbield-6.6.3-1.0.0`** (NXP `meta-imx`; Yocto **nanbield** 4.3, *not* scarthgap) | `nxp-imx/meta-imx` branch list; LF6.6.3 = nanbield |
| Vivante/OVX userspace | ✅ **`imx-gpu-viv-6.4.11.p2.4-aarch64-b07999b.bin`** — PV `6.4.11.p2.4`, srcrev `b07999b` (== board galcore `6.4.11.p2.745085`) | recipe `imx-gpu-viv_6.4.11.p2.4-aarch64.bb`; sha256 `ba86656c…2807` verified |
| TFLite/delegate | `tensorflow-imx` / `tflite-vx-delegate-imx` / `tim-vx-imx` @ `nanbield-6.6.3-1.0.0` | prebuilt from NXP rootfs (TFLite can't bazel-build on-board) |

> **DONE:** the Vivante half is fetched, sha256-verified, and extracted on the board at
> `~/npu/imx-gpu-viv-6.4.11.p2.4-aarch64-b07999b/gpu-core/usr/` — includes `libOpenVX.so.1.3.0`,
> `libVSC/libGAL/libCLC/libArchModelSw/libNNArchPerf`, the i.MX8MP NPU binaries
> (`mx8mp/libNN*Binary-evis2.so`), CL headers, but **no OpenVX `VX/*.h` headers** (tim-vx bundles those).
> Download URL: `https://www.nxp.com/lgfiles/NMG/MAD/YOCTO/imx-gpu-viv-6.4.11.p2.4-aarch64-b07999b.bin`
> (open, no login; board can't resolve nxp.com via its Tailscale MagicDNS — fetch on a normal host + scp).

> CompuLab's **Debian** image deliberately omits this stack (that's why the board has none).
> The matching binaries live in the **Yocto** `imx-image-full` rootfs built from the same BSP.

## Phase 0 — Confirm the exact versions

✅ **DONE (board side):** galcore = **`6.4.11.p2.745085`** (`/sys/kernel/debug/gc/version`),
`CONFIG_MXC_GPU_VIV=y` (builtin), `CONFIG_DRM_ETNAVIV=m` (graphics only, not NPU). `/dev/galcore` +
`/dev/dri/renderD128` (`40000000.mix_gpu_ml`) present. **This is the ABI contract.**

Remaining (no board access needed) — pin CompuLab's BSP commit and the NXP recipe versions:

```bash
# CompuLab BSP → NXP manifest
git clone -b scarthgap https://github.com/compulab-yokneam/meta-bsp-imx8mp
grep -ri "lf-6.6.3\|imx-manifest\|DISTRO_VERSION\|nxp" meta-bsp-imx8mp/   # confirm lf-6.6.3_1.0.0

# NXP recipe versions for that tag (the exact imx-gpu-viv / tflite versions)
# meta-imx @ scarthgap-6.6.3-1.0.0:
#   meta-imx-bsp/.../kernel-module-imx-gpu-viv_6.4.11.pX.Y.bb
#   meta-imx-bsp/.../imx-gpu-viv_6.4.11.pX.Y.bb
#   meta-ml/.../tensorflow-lite_*.bb , tflite-vx-delegate_*.bb , tim-vx_*.bb
```

**Gate:** the container's `imx-gpu-viv` libs MUST be **`6.4.11.p2` build `745085`** (== the board's
galcore). Pick the LF tag whose `kernel-module-imx-gpu-viv_*.bb` / `imx-gpu-viv_*.bb` resolves to
exactly that `PV`+`SRCREV`. If the scarthgap recipe shows a different build (e.g. p1.0 or p3.0), it's
the wrong tag — find the one that pins p2.745085 (likely `lf-6.6.3_1.0.0`) before building anything.

## On-device build budget (measured 2026-06-04) — this constrains everything

- **Disk:** `/` 92% full, **2.2 GB free**, single 29 GB eMMC, **no external storage**, Docker root on `/`.
  `docker system df` shows ~3.8 GB reclaimable (unused images) → `docker system prune` frees to ~6 GB
  *without* touching the running HA/Grafana/Greptime stack.
- **RAM:** 3.5 GB total, ~2 GB free, **no swap.**

Two hard consequences:
1. **No on-device Yocto/bitbake** (needs tens of GB) — `imx-kbuild` cannot do a full BSP build here.
2. **No on-device bazel build of TensorFlow Lite** (needs ≫2 GB RAM + GBs disk + swap). `libtensorflow-lite.so`
   **must be a prebuilt aarch64 binary.**

⇒ The path is **prebuilt aarch64 binaries + tiny native compiles**, with `imx-kbuild` (the user's most
accessible build container) used as a **glibc-matched build/run environment**, not a Yocto builder.
The board *is* aarch64, so every compile below is native — **no cross-compilation needed.**

## Phase 1 — Assemble the matching userspace (board-native, prebuilt-first)

Pre-req: `docker system prune` (frees ~3.8 GB) and/or attach a USB/NVMe and point Docker `data-root` at it.

| Component | How (no bitbake/bazel) | Size/cost |
|---|---|---|
| Vivante/OVX libs (`libGAL/libOpenVX/libVSC/libCLC/libOpenCL/libNNVXCBinary/libnnrt…`) + galcore firmware | Fetch the exact self-extractor **`imx-gpu-viv-6.4.11.p2-745085.bin`** from NXP FSL mirror (`${FSL_MIRROR}` in `Freescale/meta-freescale` `recipes-graphics/imx-gpu-viv/imx-gpu-viv-6.inc`, typically `https://www.nxp.com/lgfiles/NMG/MAD/YOCTO/`) → `sh …bin --auto-accept` → take the aarch64 `gpu-core/usr/lib` tree (wayland variant). **No build — binary blob.** | ~tens of MB |
| `libtensorflow-lite.so` (C-API) | **Prebuilt only.** Extract from an NXP **lf-6.6.3 prebuilt rootfs** (`imx-image-full` .wic from NXP's release page — loop-mount, copy `/usr/lib/libtensorflow-lite.so*`), or any apt/deb that carries the matching 2.x build (the i.MX 95 board had `libtensorflow-lite2.19.0` via apt — see [[imx95_baseline]]). Do **not** bazel-build on this board. | download-gated |
| `libtim-vx.so` | Native `cmake` build from `nxp-imx/tim-vx-imx` @ the lf-6.6.3 tag, against the gpu-viv OpenVX headers. C++, fits ~2 GB RAM. Do it inside `imx-kbuild`. | small build |
| `libvx_delegate.so` | Native build from `nxp-imx/tflite-vx-delegate-imx` @ lf-6.6.3, links tflite + tim-vx + OpenVX. | small build |
| `libfacex_npu.so` | Our own — `gcc` from `src/backend_tflite.c` (see Phase 2), trivial. | seconds |

> If `libtim-vx`/`libvx_delegate` fight the build, the fallback is to pull them **prebuilt** from the same
> NXP rootfs as `libtensorflow-lite` (they're all in `imx-image-full`) — then nothing is built on-device
> except `libfacex_npu.so`. That's the lowest-risk option given the RAM/disk budget; prefer it.

## Phase 2 — Build the Docker image

Base on a **glibc-matched** layer. The gpu-viv + eIQ libs are built against Yocto scarthgap glibc (2.39);
Debian bookworm is glibc 2.36 → **prefer running them in a scarthgap-glibc container** (the `imx-kbuild`
base, or a slim rootfs imported from the NXP prebuilt image) rather than on the bare Debian host.
```dockerfile
# Derived from the imx-image-full rootfs tarball (docker import), or FROM a published
# nxp/imx eIQ arm64 image if one matches lf-6.6.3_1.0.0.
FROM imx8mp-eiq:lf-6.6.3_1.0.0
# FaceX NPU lib, built against the in-image TFLite:
COPY src/backend_tflite.c include/ third_party/tflite_c/include/ /build/
RUN gcc -O3 -fPIC -DFACEX_BACKEND_TFLITE -I/build/include -I/build/third_party/tflite_c/include \
        -mcpu=cortex-a53 -shared -o /usr/lib/libfacex_npu.so /build/src/backend_tflite.c \
        -ltensorflow-lite -ldl -lm -lpthread     # TFLITE_LIBNAME=tensorflow-lite
COPY imx_npu_compile_test facex-bench-npu /usr/bin/
```
Build options:
- **On the board** (`docker build` natively, aarch64) — simplest, but slow + disk-heavy.
- **Cross-build** with `docker buildx build --platform linux/arm64` on the Mac/x86 host, then
  `docker save | ssh compulab@192.168.2.11 docker load` — avoids board disk/CPU pressure.

**Disk:** board `/` is 92% full (~2.4 GB free). Move Docker's data-root to roomy media
(`/etc/docker/daemon.json: {"data-root": "/path/on/big/mount"}`, restart docker) or load the image
onto external storage. The eIQ image is ~hundreds of MB.

## Phase 3 — Run with NPU device passthrough + validate the delegate

```bash
docker run --rm \
  --device /dev/galcore --device /dev/dri/renderD128 --device /dev/dri/card0 \
  -e LD_LIBRARY_PATH=/usr/lib \
  -v $PWD/models:/models  imx8mp-facex-npu  bash
```
Validate in this order (cheapest first):
1. **Vivante init**: a stock NXP tool, e.g. TFLite `benchmark_model
   --graph=/models/mobilenet_v1_*_quant.tflite --external_delegate_path=/usr/lib/libvx_delegate.so`
   → expect non-zero delegated partitions + a warm-up of several seconds (graph compile). 0 ops or a
   hang here = version mismatch → revisit Phase 0/1.
2. **FaceX API**: `imx_npu_compile_test /models/edgeface_xs_int8.tflite` → expect
   `active delegate: vx` + non-zero nodes delegated.

Permissions: `/dev/galcore` is `crw------- root root`; the container runs as root by default, so
`--device` passthrough works. (No SELinux on this Debian; no extra caps needed.)

## Phase 4 — Model (host-side, no board needed)

8M Plus ingests **plain INT8** — **no Vela, no neutron-converter** (the easiest of the three i.MX
targets; contrast with [[imx95_baseline]]).
```bash
python tools/onnx_to_tflite.py --int8 ... -> weights/edgeface_xs_int8.tflite
# Calibrate with 100–200 aligned 112×112 face crops (same sampling as imx95 for cross-board parity).
```
Sanity-check on host with XNNPACK before shipping to the board.

## Phase 5 — Bench + hybrid pipeline

```bash
# inside the container, on the board:
facex-bench-npu --embed /models/edgeface_xs_int8.tflite --iters 200 --warmup 20 --format csv \
  > imx8mp_npu.csv
facex-bench-npu --embed /models/edgeface_xs_int8.tflite --delegate xnnpack ... # CPU-TFLite floor
```
- Append the `vx` row to `docs/bench/imx8mp_baseline.csv` next to the existing CPU rows.
- Profile op residency (TFLite `--enable_op_profiling` / vx delegate verbose). ConvNeXt `LayerNorm`
  / `GELU` are the usual CPU-residual suspects; if >30 % of latency falls back to A53, that's the
  signal for a `BatchNorm+ReLU6` model rewrite (separate effort).
- **Hybrid pipeline**: CPU detect (`libfacex.a` — runs fine on the Debian host *or* in-container) +
  NPU embed (`libfacex_npu.so`, container). Measure end-to-end p50/p95/p99 vs the all-CPU ~120 ms.

## Phase 6 — Productize + docs

- The deliverable is the **container image** + a `docker run` recipe (add a `tools/run_npu_docker.sh`).
- Add a native `imx8mp-npu` Make target (paralleling `imx8mp-cpu`) that builds `libfacex_npu.so`
  against the in-container/staged TFLite — replaces the Yocto-SDK-only `imx8mp` target for this board.
- Flip the NPU row in `docs/coverage_matrix.md` from "blocked" to validated (EVK rev + BSP + p50).
- Update `imx8mp_baseline` memory + `docs/imx_npu.md` once numbers land.

## Risks & mitigations

- **Version mismatch (the #1 risk):** mitigated by deriving userspace from the *same* lf-6.6.3_1.0.0
  as the kernel. Phase-0 gate enforces it. If CompuLab's scarthgap kernel ≠ LF6.6.3 userspace,
  rebase the userspace tag (don't rebuild the kernel — see `docs/kernel-rebuild.md` only as last resort).
- **glibc mismatch:** Yocto libs + Debian base can clash → prefer Route A (Yocto rootfs as the
  container base).
- **Disk (2.4 GB free):** cross-build + `docker load`, and move Docker data-root to external media.
- **CompuLab scarthgap "not officially released yet":** the BSP is in flux — pin a specific commit
  SHA in Phase 0 and record it in the baseline memory.
- **Thermal:** report NPU numbers at steady state (post warm-up), not cold compile.
- **Weights license:** EdgeFace-XS is CC BY-NC-SA 4.0 — fine for eval, not for commercial example
  artifacts. Matters more for OEM conversations on embedded.

## What this plan does NOT do

- No NPU **detector** (stays `-ENOSYS`; hybrid CPU-detect is the recommended path).
- No EdgeFace architecture rewrite (op-residency fix is a separate effort, gated on Phase-5 profiling).
- No kernel rebuild unless Phase 0 proves an unavoidable version gap.
```
