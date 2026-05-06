# FaceX — Fast face embedding library
# 3ms inference, 7MB binary, zero dependencies
#
# Build:  make
# Test:   make example && ./facex-example
#         make test     (golden embedding test)
#         make bench    (synthetic latency benchmark)
#
# Auto-detects host arch:
#   x86-64 Linux/macOS/Windows  → AVX2 (+ AVX-512 + VNNI when present)
#   arm64 macOS / Linux         → NEON kernels, links gemm_stub +
#                                  threadpool_pthread, defines FACEX_NO_INT8

CC ?= cc
LDFLAGS = -lm -lpthread

UNAME_S := $(shell uname -s 2>/dev/null)
UNAME_M := $(shell uname -m 2>/dev/null)

ifeq ($(OS),Windows_NT)
  ARCH := x86_64
else ifneq (,$(filter arm64 aarch64,$(UNAME_M)))
  ARCH := arm64
else
  ARCH := x86_64
endif

ifeq ($(ARCH),arm64)
  # Apple Silicon / generic AArch64
  CFLAGS = -O3 -funroll-loops -DFACEX_NO_INT8
  ifeq ($(UNAME_S),Darwin)
    CFLAGS += -mcpu=apple-m1
  else
    CFLAGS += -march=armv8-a+simd
  endif
  GEMM_SRC = src/gemm_stub.c
  THREADPOOL_SRC = src/threadpool_pthread.c
else
  # x86-64 path (original)
  CFLAGS = -O3 -march=native -mfma -funroll-loops
  AVX512 := $(shell $(CC) -mavx512f -dM -E - < /dev/null 2>/dev/null | grep -c AVX512F)
  ifeq ($(AVX512),1)
    CFLAGS += -mavx512f -mavx512vnni -mprefer-vector-width=512
  endif
  GEMM_SRC = src/gemm_int8_4x8c8.c
  ifeq ($(UNAME_S),Darwin)
    THREADPOOL_SRC = src/threadpool_pthread.c
  else
    THREADPOOL_SRC = src/threadpool.c
  endif
endif

ifeq ($(OS),Windows_NT)
  LDFLAGS += -lsynchronization
  EXT = .exe
endif

SRCS = src/facex.c src/transformer_ops.c $(GEMM_SRC) $(THREADPOOL_SRC)

.PHONY: all clean example lib cli encrypt test bench detect-lib \
        bench-camera bench-camera-debug bench-camera-profile

all: lib cli detect-lib

# Static library
lib: libfacex.a

libfacex.a: $(SRCS) src/detect.c src/align.c src/weight_crypto.c
	$(CC) $(CFLAGS) -Iinclude -DFACEX_LIB -c src/facex.c -o facex.o
	$(CC) $(CFLAGS) -Iinclude -c src/transformer_ops.c -o transformer_ops.o
	$(CC) $(CFLAGS) -Iinclude -c $(GEMM_SRC) -o gemm.o
	$(CC) $(CFLAGS) -Iinclude -c $(THREADPOOL_SRC) -o threadpool.o
	$(CC) $(CFLAGS) -Iinclude -c src/detect.c -o detect.o
	$(CC) $(CFLAGS) -Iinclude -c src/align.c -o align.o
	$(CC) $(CFLAGS) -Iinclude -c src/weight_crypto.c -o weight_crypto.o
	ar rcs $@ facex.o transformer_ops.o gemm.o threadpool.o detect.o align.o weight_crypto.o
	@rm -f facex.o transformer_ops.o gemm.o threadpool.o detect.o align.o weight_crypto.o
	@echo "Built libfacex.a ($(ARCH))"

# Standalone CLI (for Go subprocess / testing)
cli: facex-cli$(EXT)

facex-cli$(EXT): src/edgeface_engine.c src/transformer_ops.c $(GEMM_SRC) $(THREADPOOL_SRC) src/weight_crypto.c src/detect.c src/align.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	@echo "Built facex-cli$(EXT) ($(ARCH))"

# Example program
example: facex-example$(EXT)

facex-example$(EXT): examples/example.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# Encryption tool
encrypt: facex-encrypt$(EXT)

facex-encrypt$(EXT): src/weight_crypto.c
	$(CC) $(CFLAGS) -DWEIGHT_CRYPTO_MAIN -o $@ $< $(LDFLAGS)

# Golden test
test: golden-test$(EXT)
	@echo "Running golden test..."
	@./golden-test$(EXT) data/edgeface_xs_fp32.bin

golden-test$(EXT): tests/golden_test.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# Unified latency bench. Same source / same output schema across every
# build flavour — see scripts/bench_all.sh for the sweep that produces
# a single comparison table.
bench: facex-bench$(EXT)

facex-bench$(EXT): tools/bench.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# macOS camera benchmark (Swift, AVFoundation). Requires Xcode CLT swiftc.
# The build script handles the swiftc invocation + bridging header.
bench-camera: libfacex.a
	@command -v swiftc >/dev/null || { echo "swiftc not found — install Xcode Command Line Tools"; exit 1; }
	@bash tools/build_bench_camera_mac.sh release

bench-camera-debug: libfacex.a
	@command -v swiftc >/dev/null || { echo "swiftc not found — install Xcode Command Line Tools"; exit 1; }
	@bash tools/build_bench_camera_mac.sh debug

bench-camera-profile: libfacex.a
	@command -v swiftc >/dev/null || { echo "swiftc not found — install Xcode Command Line Tools"; exit 1; }
	@bash tools/build_bench_camera_mac.sh profile

# Detector static library
detect-lib: libdetect.a

libdetect.a: src/detect.c include/detect.h
	$(CC) $(CFLAGS) -Iinclude -c src/detect.c -o detect.o
	ar rcs $@ detect.o
	@rm -f detect.o
	@echo "Built libdetect.a"

clean:
	rm -f libfacex.a libdetect.a \
	      facex-cli$(EXT) facex-example$(EXT) facex-encrypt$(EXT) \
	      golden-test$(EXT) facex-bench$(EXT) facex-camera-bench *.o
