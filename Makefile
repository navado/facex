# FaceX — Fast face embedding library
# 3ms inference, 7MB binary, zero dependencies
#
# Build:  make
# Test:   make example && ./facex-example

CC ?= gcc
CFLAGS = -O3 -march=native -mfma -funroll-loops
LDFLAGS = -lm -lpthread

# Detect AVX-512
AVX512 := $(shell $(CC) -mavx512f -dM -E - < /dev/null 2>/dev/null | grep -c AVX512F)
ifeq ($(AVX512),1)
  CFLAGS += -mavx512f -mavx512vnni -mprefer-vector-width=512
endif

ifeq ($(OS),Windows_NT)
  LDFLAGS += -lsynchronization
  EXT = .exe
endif

SRCS = src/facex.c src/transformer_ops.c src/gemm_int8_4x8c8.c src/threadpool.c

.PHONY: all clean example lib cli encrypt test

all: lib cli

# Static library
lib: libfacex.a

libfacex.a: $(SRCS)
	$(CC) $(CFLAGS) -DFACEX_LIB -c src/facex.c -o facex.o
	$(CC) $(CFLAGS) -c src/transformer_ops.c -o transformer_ops.o
	$(CC) $(CFLAGS) -c src/gemm_int8_4x8c8.c -o gemm_int8_4x8c8.o
	$(CC) $(CFLAGS) -c src/threadpool.c -o threadpool.o
	ar rcs $@ facex.o transformer_ops.o gemm_int8_4x8c8.o threadpool.o
	@rm -f *.o
	@echo "Built libfacex.a"

# Standalone CLI (for Go subprocess / testing)
cli: facex-cli$(EXT)

facex-cli$(EXT): src/edgeface_engine.c src/transformer_ops.c src/gemm_int8_4x8c8.c src/threadpool.c src/weight_crypto.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	@echo "Built facex-cli$(EXT)"

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

clean:
	rm -f libfacex.a facex-cli$(EXT) facex-example$(EXT) facex-encrypt$(EXT) golden-test$(EXT) *.o
