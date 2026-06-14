# i.MX 8M Plus — Proper Support Plan

Move FaceX on i.MX 8M Plus from "compiles, untested" to "validated on hardware, on the bench dashboard, and a recommended target."

## Where we are today

- `make imx8mp SDK=…` builds `libfacex_npu.so` for the board (Cortex-A53, `armv8-a+crc`). Source is the shared `src/backend_tflite.c`; tuning is the only delta from `imx93`/`imx95`.
- VxDelegate (`libvx_delegate.so`) is the runtime-selected delegate; expected `facex_npu_active_delegate(fx) == "vx"`.
- Detector path returns `-ENOSYS`; deployment is hybrid (CPU detect via `libfacex.a` + NPU embed via `libfacex_npu.so`).

## Reality on the CompuLab board (validated 2026-06-04)

The actual hardware on hand is **not an NXP Yocto EVK** — it's a **CompuLab IOT-GATE-IMX8PLUS** (`compulab@192.168.2.11`) running **Debian 12 (bookworm)**, kernel `6.6.3`, 4× Cortex-A53, 3.5 GiB LPDDR4, native gcc 12.2.0. That changes the bring-up plan materially:

- **CPU path is validated on hardware.** A native `make` (now also `make imx8mp-cpu`, A53-tuned) builds `libfacex.a` + `facex-cli` + `facex-bench` on-device with zero external deps. `make test` PASSES — NaN=0, self-similarity 1.000, self-consistency diff 0, different-input similarity **0.7864** (bit-identical to the Mac NEON build, so no panel-pack corruption).
- **Baseline numbers committed** to `docs/bench/imx8mp_baseline.csv`. Embed (EdgeFace-XS, NEON): median **58.9 ms**, p99 **61.5 ms**. Detector-only (e2e, no face): median **60.6 ms**. All-CPU hybrid (detect + embed, 1 face) ≈ **120 ms**.
- **Row-parallel MLP wired up.** `_mlp_rows` in `src/edgeface_engine.c` was dead code (defined, never dispatched) — the engine spawned 4 workers via `tp_init(4)` but ran the whole forward pass on one core. `convnext_block` now fans the MLP across the threadpool with `tp_parallel_for(_mlp_rows, …)`. Output is bit-identical (verified on Mac + board). Embed went 69.8 → 58.9 ms (~1.18×). The modest gain — despite ~3 cores busy — is because **EdgeFace-XS on A53 is memory-bandwidth bound** (shared LPDDR4 + Amdahl: attention/DW-conv/LN/stem/FC are still serial). This is the strongest signal yet that the **NPU, not more CPU threads, is the real win on this board.** (Bonus: the fix helps every NEON target — it also beats the i.MX 95 A55 baseline of 62.97 ms.)
- **NPU is blocked by missing userspace** — see next section. `/dev/galcore` (the VIP9000 kernel driver, char 199:0) IS present, but the entire Verisilicon/TFLite userspace is absent on this Debian image.

## Missing components & how to obtain them

> **Concrete NPU bring-up plan: see [`imx8mp_npu_bringup.md`](imx8mp_npu_bringup.md)** — Docker-hosted,
> userspace sourced from CompuLab's `meta-bsp-imx8mp` scarthgap BSP (NXP `lf-6.6.3_1.0.0`). The table
> below is the summary; the bring-up doc has the staged, command-level plan.

To go from "CPU validated" to "NPU running" on this Debian board, these pieces are missing. Listed in dependency order; each must version-match the in-kernel `galcore` driver or you get the classic "delegate dlopens but executes 0 ops" / hang.

| Component | Status on board | Where it comes from |
|---|---|---|
| `galcore` kernel driver (`/dev/galcore`) | ✅ present (char 199:0) | In-tree; already loaded |
| Verisilicon OVX userspace — `libOpenVX.so`, `libVSC.so`, `libGAL.so`, `libArchModelSw.so`, `libNNArchPerf.so`, `libNNGPUBinary*`, `libnnrt.so` | ❌ absent | NXP Yocto `imx-gpu-viv` package; or CompuLab's BSP/`apt` repo for this image |
| `libtensorflow-lite.so` (with C-API symbols) | ❌ absent | NXP `tensorflow-lite` Yocto package (the i.MX 95 board had `libtensorflow-lite2.19.0` via apt — check CompuLab's repo for the same) |
| `libvx_delegate.so` | ❌ absent | NXP `tensorflow-lite-vx-delegate` Yocto package |
| `edgeface_xs_int8.tflite` | ❌ not yet produced | Host-side via `tools/onnx_to_tflite.py` — plain INT8, **no** Vela / neutron-converter needed for 8M Plus |
| FaceX NPU header set | ✅ vendored | `third_party/tflite_c/include/` (14 headers, used by `make imx8mp`) |

**Recommended acquisition path (lowest risk → highest):**

1. **Check CompuLab's apt repo / BSP first.** The i.MX 95 board got its whole stack from `apt` (`libtensorflow-lite2.19.0`, neutron delegate) on an NXP-derived image. CompuLab ships a Debian BSP for this gateway — look for `imx-gpu-viv`, `tensorflow-lite`, and a `vx-delegate` package, plus a `-dev` for headers (the i.MX 95 image had **no** headers, hence the vendored set — expect the same here).
2. **If apt has nothing**, pull the libraries out of a matching NXP Yocto image (`fsl-image-*` for the same BSP version as this kernel, LF6.6.x) and stage them under `/usr/lib` + `ldconfig`. The driver↔userspace version match is the whole ballgame — pin `imx-gpu-viv` to the same LF6.6.x tag as the running kernel (`uname -r` → `6.6.3`).
3. **Once staged**, build with `make imx8mp` is not applicable (no Yocto SDK on a Debian box); instead build the NPU lib natively, mirroring the i.MX 95 recipe: `gcc -O3 -fPIC -DFACEX_BACKEND_TFLITE -Iinclude -Ithird_party/tflite_c/include -mcpu=cortex-a53 -shared -o libfacex_npu.so src/backend_tflite.c -ltensorflow-lite -ldl -lm -lpthread` (use `TFLITE_LIBNAME=tensorflow-lite`, add a soname symlink if the lib is `…so.2.x`). **TODO:** add a native `imx8mp-npu` Make target once the stack is confirmed, paralleling `imx8mp-cpu`.
4. **Verify** with `imx_npu_compile_test edgeface_xs_int8.tflite` — pass criteria `active delegate: vx` + non-zero nodes delegated.

Until step 1–2 land, the **shippable deployment on this board is all-CPU** (~120 ms hybrid) via `make imx8mp-cpu`.

## Goal

By the end of this work:

1. `imx_npu_compile_test` runs on an 8M Plus EVK and prints `active delegate: vx` for a converted `edgeface_xs_int8.tflite`.
2. `facex-bench-npu` produces a row that drops into `scripts/bench_all.sh`'s comparison table — same schema as host + i.MX 95.
3. `docs/coverage_matrix.md` flips 8M Plus from "syntax-only" to "validated on EVK rev. X, BSP version Y."
4. `docs/imx_npu.md` §6 ("Known limitations") loses the "hardware-untested" caveat for 8M Plus specifically.
5. A baseline number (median/p99 embed latency on VxDelegate vs XNNPACK fallback) is committed to repo so regressions are catchable.

## Phases

### Phase 0 — Toolchain & host prep (no board needed)

- Pull NXP Yocto SDK matching the EVK's BSP. Confirmed working layout: `/opt/fsl-imx-xwayland/6.6-scarthgap/`. Save the SDK version this plan validates against; mismatched BSP/SDK is the #1 cause of "delegate dlopens but executes 0 ops."
- Verify the SDK ships `libtensorflowlite_c.so` (or `libtensorflow-lite.so` with the C-API symbols exposed) and `libvx_delegate.so` under `sysroots/aarch64-poky-linux/usr/lib/`. Adjust `TFLITE_LIBNAME` if needed.
- Build `libfacex_npu.so` for 8M Plus on the host: `make imx8mp SDK=/opt/fsl-imx-xwayland/6.6-scarthgap`. Confirm clean build.

### Phase 1 — Model conversion

- Produce `weights/edgeface_xs_int8.tflite` once on the host (`tools/onnx_to_tflite.py` already exists from the `imx-npu` work). 8M Plus ingests plain INT8 — **no** Vela, **no** neutron-converter. This is the easiest of the three i.MX targets.
- Cache calibration: 100–200 aligned 112×112 face crops, identical sampling to the `imx95` calibration so int8 results are comparable across boards.
- Sanity-check on the host with XNNPACK fallback: `./imx_npu_compile_test weights/edgeface_xs_int8.tflite` should report a working delegate (xnnpack) and embed-side numerical sanity.

### Phase 2 — Bring-up (on EVK)

Follow `docs/imx_npu.md:302` checklist, 8M Plus column:

- Kernel: confirm `CONFIG_GALCORE=y` in the running kernel. `/proc/config.gz` if exposed; else `zcat /proc/config.gz | grep GALCORE`.
- Device node: `/dev/galcore` exists with rw permissions for the runtime user. Fix udev if not.
- Delegate plugin: `ldconfig -p | grep vx_delegate` shows `libvx_delegate.so` at a known path. If not, set `LD_LIBRARY_PATH=/usr/lib`.
- Firmware: VxDelegate has no separate firmware blob (the Verisilicon driver is in-tree); skip the firmware line of the checklist.

Run on the board:

```
scp libfacex_npu.so imx_npu_compile_test edgeface_xs_int8.tflite root@evk:/tmp/
ssh root@evk "cd /tmp && LD_LIBRARY_PATH=. ./imx_npu_compile_test edgeface_xs_int8.tflite"
```

Pass criteria: prints `active delegate: vx` AND non-zero `nodes delegated`. If it reports `xnnpack`, `verbose=1` in `FaceXNpuOptions` and look at stderr — usually the .so isn't on the loader path or the driver isn't loaded.

### Phase 3 — Latency baseline

- Cross-compile `facex-bench-npu` for 8M Plus (mirror the `imx95` recipe — add an `imx8mp-bench` target to the Makefile that runs `tools/bench_npu.c` against the cross-toolchain).
- Run on the board, 200 iters, 20 warmup:
  ```
  ./facex-bench-npu --embed edgeface_xs_int8.tflite --iters 200 --warmup 20 --format csv > 8mp_baseline.csv
  ```
- Also collect the XNNPACK-fallback number (`--delegate xnnpack`) on the same board so the NPU speedup is visible in the table.
- Commit `docs/bench/imx8mp_baseline.csv` so future runs catch regressions.

### Phase 4 — Hybrid pipeline integration

- On the EVK, build `libfacex.a` for the board (already supported via the existing `arm64` path with `cortex-a53` tuning — add an `imx8mp-cpu` target if convenient).
- Wire the hybrid app pattern from `docs/imx_npu.md:221`:
  - CPU detect via `libfacex.a` / `detect_run()`
  - CPU align via `align_face()`
  - NPU embed via `facex_npu_embed()`
- Measure end-to-end (detect + align + embed) p50/p95/p99. Compare to `facex_detect()` all-CPU on the same board as the speedup reference.

### Phase 5 — Documentation + matrix flip

- `docs/coverage_matrix.md`: change 8M Plus row from "syntax-only" to a real status with EVK rev + BSP version + p50 latency.
- `docs/imx_npu.md`: drop the "hardware-untested" caveat for 8M Plus only; keep it for i.MX 93/95 until those see hardware too.
- Add `docs/bench/imx8mp_baseline.csv` (one row, version-controlled, easy to re-run and diff).
- Update CLAUDE.md i.MX bullet to mention the validated board if useful.
- Auto-memory: write an `imx8mp_baseline.md` like the existing `imx95_baseline.md` so future sessions know the board's perf shape.

## Perf targets (sanity checks, not contracts)

Rough envelope, to be confirmed in Phase 3:

| Path | Expected p50 | Notes |
|---|---|---|
| `facex_npu_embed` via VxDelegate INT8 | 8–15 ms | VIP9000 + INT8 EdgeFace-XS; will trend higher if too many ops fall back to CPU |
| `facex_npu_embed` via XNNPACK on A53 | 50–90 ms | A53 quad-core, FP32 fallback; this is the floor the NPU has to beat |
| CPU `facex_detect` (YuNet) on A53 NEON | 10–25 ms | Already supported via existing arm64 build path |
| Hybrid end-to-end (1 face) | 20–35 ms | detect + align + NPU embed |

If embed p50 lands above 30 ms on VxDelegate, op coverage is the suspect — TFLite's `--profile` flag or `verbose=1` shows which nodes are CPU-residual.

## Risks & known issues

- **VxDelegate quirks.** Older NXP BSPs (anything pre-LF6.6) had VxDelegate bugs around dynamic-shape ops. If we hit unexplained correctness drift on certain inputs, the first move is BSP version. Pin to LF6.6+ to avoid known landmines.
- **Op residency.** Same risk as i.MX 95: ConvNeXt blocks have ops VxDelegate decomposes (LayerNorm, GELU). Expect a few layers to run on A53 — quantify the cost early (Phase 3, `--profile`). If it's >30% of latency, a model rewrite using `BatchNorm` + `ReLU6` is the long-term fix.
- **Detector path stays CPU.** `facex_npu_detect` returns `-ENOSYS` and we're not changing that here. If hybrid CPU detect on A53 NEON is the bottleneck (likely on multi-face frames), the right move is downsizing input or using a smaller detector, not porting it to the NPU.
- **Thermal.** Sustained 8M Plus NPU runs can hit thermal throttling on bare EVKs without a heatsink. Bench numbers should be reported at steady state (after a few seconds of warmup), not cold-start.
- **EdgeFace weights license.** CC BY-NC-SA 4.0 — fine for evaluation, not OK to bake into commercial example artifacts. Same rule as host builds; matters more here because OEM customers ask "is this shippable?" earlier on embedded.

## Why this is the right order

Phase 0–2 are all blocking — without bring-up there's nothing to bench. Phase 3 is the first thing that can regress, so it goes immediately after. Phase 4 is on the critical path for any real customer ("can I do faces on 8M Plus?") and is the smallest delta once Phase 3 works. Phase 5 is the cheapest step and the one most likely to be skipped — gating "validated" on docs being updated keeps the matrix honest.

## What this plan deliberately does not do

- Doesn't try to add NPU detection. The hybrid path is the recommended deployment and is already wired; spending time on NPU detect for the marginal win isn't justified until the bench dashboard says detect is the bottleneck.
- Doesn't add CI. There's no shared 8M Plus runner; gating PRs on hardware we don't own is worse than the current "manual run on bring-up" cadence. Revisit when a hosted runner becomes available.
- Doesn't touch the EdgeFace-XS architecture. Op-residency fixes via model rewrite are a separate, larger effort and shouldn't block first-light on the board.
