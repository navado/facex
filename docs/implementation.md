# FaceX implementation details

This document records, per topic, what's actually wired in the codebase
and what's gated as future work. It supersedes the earlier forward-
looking sprint plan and is written from the implementation's point of
view; **each top-level topic was added by its own commit**.

| § | Topic | Where |
|---|---|---|
| 1 | Benchmark infrastructure | This commit |
| 2 | Apple Silicon / Mac perf paths | Mac commit |
| 3 | i.MX NPU library | i.MX commit |
| 4 | ESP32-P4 ESP-IDF component | ESP32 commit |

## Status legend

| | meaning |
|---|---|
| ✅ | Implemented, tested end-to-end on this host |
| 🧪 | Compiles + links + static checks pass; runtime needs hardware/SDK we don't have |
| 🛠 | Code path exists, currently a stub |
| 🚫 | Documented but not implemented in this repo |

---

## 1. Benchmark infrastructure

A single source of truth for FaceX latency numbers across build flavours,
OSes, and stages — replaces what used to be five scattered ad-hoc
benches each with its own format.

### Components

| File | Role |
|---|---|
| `tools/bench.c` | Cross-platform synthetic latency bench. Same source compiles on macOS arm64/x86, Linux aarch64, future i.MX targets. ✅ on mac-m2. |
| `tools/bench_camera_mac.swift` | Live AVFoundation camera bench — different role (capture pipeline + dispatch + display), Mac-only. ✅ on mac-m2 (29 fps end-to-end). |
| `tools/build_bench_camera_mac.sh` | swiftc invocation + bridging-header generation for the camera bench. Auto-detects optional libfacex symbols (Accelerate / Core ML) and links matching frameworks. ✅. |
| `scripts/bench_all.sh` | Sweeps build-flag combinations, runs `facex-bench` against each, emits a single Markdown or CSV table. ✅. |
| `scripts/test_all.sh` | Runs every test that's runnable on this host. Topic-specific checks are added per commit. ✅ — this commit registers `make bench` checks. |
| `docs/benchmarking.md` | "Which tool answers which question" matrix, CSV schema, recipe for combining engine + camera output. |
| `docs/coverage_matrix.md` | Compiles? Static-checks? Runs end-to-end? per (target × backend × build flag). Filled from real `scripts/test_all.sh` output. |

### Output schema (shared by `tools/bench.c` and `bench_camera_mac.swift --summary`)

```
label,compiled,active,stage,iters,min_ms,median_ms,mean_ms,p95_ms,p99_ms,e2e_face
```

The `compiled` column reflects which `FACEX_HAVE_*` macros were defined
at compile time; `active` reflects what runtime probes (`sysctlbyname`
on macOS, `AT_HWCAP2` on Linux) report at first call. Subsequent topic
commits register new compile-time macros (`FACEX_HAVE_ACCELERATE`,
`FACEX_HAVE_SME`, `FACEX_HAVE_COREML`, `FACEX_BACKEND_TFLITE`) and they
appear in this column without the bench tool changing.

### Build foundation added by this commit

To make `make` succeed on Apple Silicon and AArch64 Linux at all, this
commit also lays down:

- **Arch detection in `Makefile`.** `uname -m` selects the arm64 vs
  x86_64 path. On arm64 we link `src/gemm_stub.c` (because the existing
  `gemm_int8_4x8c8.c` is x86-only) and `src/threadpool_pthread.c`
  (because `src/threadpool.c` uses Linux `futex`/Windows `WaitOnAddress`
  — neither portable).
- **`FACEX_NO_INT8` build flag.** Defined automatically on arm64.
  Wraps the engine's INT8 weight-packing block in
  `src/edgeface_engine.c` so `mm->packed` stays NULL and the matmul
  dispatch falls cleanly through to the FP32-packed path.
- **NEON kernels for `matmul_fp32_packed{,_bias,_bias_gelu}` in
  `src/transformer_ops.c`.** Hand-written `vfmaq_f32`-based 4×8 panel
  kernels. Same packed format as the AVX2 path. Output is byte-identical
  to scalar within ULP.
- **Column-panel-aware scalar fallbacks** for the same three matmul
  functions. The previous scalar `#else` branch fed packed B into
  `matmul_fp32` (which expects row-major B) — silently wrong on every
  non-x86 host.
- **`src/threadpool_pthread.c`.** Pthread + condvar pool replacing the
  `futex`/`WaitOnAddress` impl. Used on macOS today; will also be needed
  by every other ARM target.

### Verification

```bash
scripts/test_all.sh    # 100% checks runnable on this host
scripts/bench_all.sh   # produce comparison table across build flavours
```

Latest result on **mac-m2** (default build, post-Bench-foundation):

```
embed     median 4.5–4.7 ms (NEON FP32 packed)
e2e       median 8.4 ms     (detect + align + embed, 1 face)
fps       29-30 (camera-limited at sessionPreset .vga640x480)
```

Apple-specific perf paths that beat NEON (Accelerate AMX, SME, Core ML
ANE) are added in the Mac commit — not part of the Bench foundation.

---

## 2. Apple Silicon / Mac perf paths

Default `make` on Apple Silicon already runs the NEON kernels from the
Bench foundation. This section adds the **Apple-specific** acceleration
paths — they're opt-in build flags, never the default, so the same source
keeps shipping to anyone with any Mac.

### What's added

| File | Role |
|---|---|
| `src/cpu_features.{h,c}` | Runtime probe for `FEAT_SME` / `FEAT_SME2` via `sysctlbyname`; cached, atomic, no external deps. Used by SME dispatcher; designed to be reused by future runtime probes (FP16, BF16, dotprod). |
| `src/transformer_ops_sme.c` | `__arm_locally_streaming __arm_new("za")` `matmul_fp32_packed` using `FMOPA` outer products into ZA tile 0. Pre-transposes A row tile (gather not allowed in streaming mode). Returns -1 on shapes it refuses (M < SVL/4 or K > 4096). Self-check at first matmul disables SME on output divergence > 1e-3. |
| `src/backend_accelerate.c` | `cblas_sgemm` wrapper. Unpacks column-panel B to row-major, dispatches via Accelerate; AMX wins for M ≥ 4 and M·K·N ≥ 4096, otherwise falls back to NEON. Self-check at init: cblas vs scalar within 1e-4 relative. |
| `src/backend_coreml.m` | Objective-C bridge (ARC) loading `.mlpackage` via `MLModel`. Configurable `compute_units` hint (ALL / CPU+GPU / CPU-only / CPU+ANE). L2-normalises output so cosine sim matches CPU backend. |
| `include/facex_coreml.h` | Public C API for the Core ML bridge. |
| `tests/test_mac.c` | Smoke test: load weights, embed sanity, determinism, self/cross similarity, latency stats, end-to-end detect. Now also reports compiled-in vs runtime-active backends. |
| `tools/export_coreml.py` | ONNX → `.mlpackage` via `coremltools.convert(convert_to="mlprogram")` + INT8 palettization (`coremltools.optimize.coreml.palettize_weights`). Required to feed the Core ML bridge. |
| `docs/mac.md` | Full Mac story — build modes, runtime fallback chain, permissions, perf reference table. |

`src/transformer_ops.c` gains a dispatcher block at the top of
`matmul_fp32_packed`:

```
Accelerate (AMX, M≥4 K·N≥4096) → SME (M4+) → NEON / AVX2 / scalar
```

Each opt-in dispatch is gated at compile time (`FACEX_HAVE_ACCELERATE`,
`FACEX_HAVE_SME`) AND at runtime (cached self-check + capability probe).

### Build matrix

| Make invocation | What gets compiled in |
|---|---|
| `make` | NEON only (default; portable across M1-M5) |
| `make ACCELERATE=1` | + AMX path via `cblas_sgemm` |
| `make SME=1` | + M4+ SME path via FMOPA |
| `make COREML=1` | + Core ML / ANE bridge (.mlpackage loader) |
| `make ACCELERATE=1 SME=1 COREML=1` | all three; dispatcher chains them |
| `make mac-universal` | fat arm64 + x86_64 archive |

**Critical isolation:** `-march=armv9-a+sme` is applied PER-FILE (only to
`transformer_ops_sme.c`), not globally. Without that, clang auto-vectorizes
plain C in `transformer_ops.c` using SVE/SME instructions that trap on
M1/M2/M3. Verified: `transformer_ops.o` contains zero `rdvl`/`smstart`/
`fmopa`; `transformer_ops_sme.o` contains the expected `fmopa za0.s`.

### Measured on mac-m2

| Build | Embed median | E2E | Status |
|---|---:|---:|---|
| Default (NEON) | ~4.6 ms | ~8.4 ms | ✅ tested |
| `ACCELERATE=1` (AMX) | ~3.5 ms (-22%) | ~7.5 ms (-13%) | ✅ tested |
| `SME=1` on M2 | ~4.6 ms | ~8.4 ms | ✅ tested (SME inert; runtime probe = 0) |
| `SME=1` on M4 | est. ~1.5 ms | est. ~5 ms | 🚫 needs M4 hardware (self-check guards correctness) |
| `COREML=1` w/ real `.mlpackage` | est. ~0.8 ms | — | 🚫 needs ONNX export (tools/export_coreml.py is in tree) |
| `mac-universal` (arm64+x86_64) | n/a | n/a | ✅ archive built; per-slice asm verified |

Same embedding bytes regardless of backend choice — `||emb||²=0.0756`,
self-similarity 1.0000, identical bbox. Backend ordering can shift the
LSB by ~ULP; the self-check gates anything worse than 1e-4 relative.

### Status by row

- ✅ **NEON** (foundation; lives in Bench commit)
- ✅ **Accelerate / AMX** — full e2e test on M2
- ✅ **SME** — compiles + emits real `fmopa`; runtime self-check guards
  M4 correctness; not directly hardware-tested
- 🧪 **Core ML** — bridge compiles + links; missing-`.mlpackage` smoke
  passes; ANE dispatch not validated end-to-end (needs the ONNX export)
- ✅ **Universal binary** — both slices contain real arch-specific code

---

<!-- Subsequent topic sections appended by their respective commits. -->
