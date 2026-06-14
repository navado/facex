# i.MX 8M Plus — performance profiling & optimization

Why the CPU barely scales with cores, why EdgeFace-XS INT8 on the NPU is "only" 5.6×
(vs MobileNet's 14.4×), and what to do about it. All numbers measured 2026-06-14 on the
CompuLab IOT-GATE-IMX8PLUS.

## CPU: memory-bandwidth bound (core-scaling)

`facex-bench` embed under `taskset` (EdgeFace-XS, NEON, row-parallel MLP):

| Cores | median ms | throughput (inf/s) | vs 1-core |
|--:|--:|--:|--:|
| 1 | 70.3 | 14.2 | 1.00× |
| 2 | 59.4 | 16.8 | **1.18×** |
| 3 | 59.6 | 16.8 | 1.18× |
| 4 | 58.2 | 17.2 | 1.21× |

The plateau from 2→4 cores is the signature of **shared-DRAM-bandwidth saturation**: two A53
cores already max out the single LPDDR4 controller, so cores 3–4 just stall (they look "busy"
in `top` — spinning on cache-miss stalls, not doing work). Root cause is low **arithmetic
intensity**: embed is ~100 M MACs but streams the full ~7 MB of FP32 weights per inference,
which blows past the A53 caches. Adding cores adds compute, not bandwidth. Secondary cap
(Amdahl): only the MLP is threaded — XCA attention, DW convs, LayerNorm, stem, FC, and the
small-spatial (`HW<64`) MLP stages stay serial. **Implication: more CPU threads won't help;
the lever is bytes moved (INT8 weights, cache-blocking), not cores.**

## NPU: profile of EdgeFace-XS INT8 (VxDelegate, `VIV_VX_PROFILE=1`)

- **Full delegation, no CPU fallback** — TFLite op-profiling shows the entire graph as one
  `Vx Delegate` node.
- **~966 operations per inference** — EdgeFace's LayerNorm, tanh-GELU, and XCA cross-covariance
  attention decompose into a long tail of tiny elementwise/transpose/reshape ops.
- **97.6% of GPU cycles are *idle*** (2.48 B idle / 2.54 B total across the profiled runs). The
  NPU spends almost all its time stalled — per-op dispatch/sync overhead across the fragmented
  graph + waiting on memory — not computing.
- **On-chip SRAM unused: `AXI_SRAM_READ/WRITE_BANDWIDTH = 0`.** All ~73 MB of reads go to DDR;
  the VIP9000's on-chip SRAM scratchpad is never used, so every layer streams weights/activations
  from DRAM.

That combination — many tiny ops + DDR streaming + no SRAM tiling — is exactly why EdgeFace gets
5.6× while MobileNetV1 (big, regular, SRAM-tileable convs that keep the NPU busy) gets 14.4× on
the same hardware.

## Optimizations (ranked)

### 1. Graph-binary cache / "preload" — validated, fixes cold-start
The first inference compiles the graph + lays out weights (**42.8 s**). Caching the compiled
network binary cuts that to **4.0 s** (~10.6×); steady-state latency is unchanged.

```sh
-e VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 -e VIV_VX_CACHE_BINARY_GRAPH_DIR=/persistent/cache
# or the vx_delegate options: allowed_cache_mode=1, cache_file_path=/persistent/cache/efx.nb
```
Pre-compile once, persist the `*.nb` (3.6 MB), ship it; every process start then loads it instead
of recompiling. For a **long-running inference process** the compile is paid once at startup and
amortizes to zero — so also: **keep the model loaded** (don't spawn a process per inference).

### 2. Make the model NPU-friendly — biggest steady-state win (needs retraining)
The 97.6%-idle / 966-op profile says EdgeFace is dying by a thousand tiny ops. Cut op count and
keep the NPU busy:
- **LayerNorm → BatchNorm** (foldable into the preceding conv; no per-element rsqrt/div op chain).
- **GELU → ReLU6 / hardswish** (single op vs the tanh-approx's mul/add/tanh chain).
- **Rework / avoid the XCA cross-covariance transposes** (the attention is transpose-heavy).
This is the path to MobileNet-like utilization. It's a model change (retrain — e.g. an
EdgeFace-Nano or a quant/NPU-friendly variant), so it's the largest effort but the real lever.

### 3. Enable on-chip AXI SRAM — cut DDR streaming (small win on this SoC)
`AXI_SRAM_READ/WRITE_BANDWIDTH = 0` in the profile → the NPU stages nothing in an external
on-chip SRAM scratchpad; all ~18 MB/inference of reads hit DDR.

**Current state on this board** (`/sys/module/galcore/parameters/`): the SRAM pools are
unconfigured — `sRAMSizes = 0,0,…`, `externalSize = 0,0`, `extSRAMSizes = 0`,
`contiguousSize = 0xFFFFFFFF` (NPU memory comes from CMA/DDR). The only reserved on-chip SRAM is
`ocram@900000` (448 KiB, `nomap non-reusable`) — already claimed (ATF/suspend), not given to the NPU.
And **galcore is built into the kernel** (`CONFIG_MXC_GPU_VIV=y`), so params can't be set with
`modprobe` — they must come from the kernel command line or device tree.

**How to enable:**
1. Reserve an on-chip SRAM range for the NPU in the device tree (`reserved-memory` node) and point
   the VIP/`gpu3d` node at it (NXP-supported route).
2. Or pass it to galcore on the kernel cmdline (quick test):
   ```
   galcore.extSRAMSizes=0x100000 galcore.extSRAMBases=<phys_addr_of_reserved_sram>
   # optionally galcore.sRAMSizes=<bytes> for the VIP per-core SRAM
   ```
3. Rebuild kernel/DTB (galcore is builtin — see `docs/kernel-rebuild.md`), deploy, **reboot**.
4. Verify: re-run with `VIV_VX_PROFILE=1` and confirm `AXI_SRAM_*_BANDWIDTH > 0` and that
   `DDR_READ_BANDWIDTH` / idle cycles drop.

**Expected effect — modest, likely single-digit %, for *this* model (not yet measured):**
- The dominant cost here is **per-op dispatch/sync overhead** (97.6% idle across ~966 tiny ops),
  which an SRAM scratchpad does **not** address — it only reduces the *memory-stall* slice of the
  idle time. So the headroom from SRAM alone is bounded by that slice, not the whole 97.6%.
- The i.MX8MP has **very little spare on-chip SRAM** (OCRAM is 448 KiB and already taken; there is
  no large dedicated NPU AXI-SRAM like higher-end i.MX). A ~256 KiB–1 MiB tile can stage some
  intermediate activations but can't hold the working set.
- EdgeFace-XS's **intermediate tensors are small** (≤ 112×112×32 early, shrinking after), so they
  largely fit in the VIP's internal SRAM already; the external-SRAM benefit is marginal. AXI-SRAM
  pays off far more for conv-heavy models with **large** feature maps (e.g. MobileNet-class).
- **Realistic estimate: ≈ 5–15% steady-state latency improvement at best on EdgeFace-XS, possibly
  negligible.** It requires a reboot + kernel/DTB rebuild on a production board for a small, model-
  dependent gain — so prioritize #1 (graph cache, validated 10.6× startup) and #2 (op-count
  reduction, the real steady-state lever) first; treat AXI-SRAM as a measure-then-keep tweak.

### 4. QAT — prerequisite for *usable* INT8 (accuracy, not speed)
Independent of the above: post-training INT8 gives cosine ~0.29 (broken). Quantization-aware
training (or mixed precision keeping LayerNorm/attention in higher precision) is required to get
a deployable recognizer. See `README.md`.

### 5. Batching / pipelining for throughput
Single-stream leaves the NPU idle between inferences. Batching or pipelining requests raises
throughput — but with 97.6% idle already coming from per-op overhead, fixing op count (#2) pays
off more than batching here.

## How to reproduce the profile
```sh
# per-layer NPU counters (idle cycles, DDR vs SRAM bandwidth):
docker run ... -e VIV_VX_PROFILE=1 facex-npu-rt:2404 \
  benchmark_model --graph=edgeface_xs_int8.tflite --num_runs=3 --warmup_runs=1 \
    --enable_op_profiling=true --external_delegate_path=/eiq/lib/libvx_delegate.so
# CPU core-scaling:
for n in 1 2 3 4; do taskset -c $(seq -s, 0 $((n-1))) ./facex-bench --stage embed --format csv; done
```
