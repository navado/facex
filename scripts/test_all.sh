#!/usr/bin/env bash
# test_all.sh — run every test that's runnable in this environment.
#
# Exits non-zero on the first failure. Prints a summary at the end so
# the coverage_matrix.md can be filled from the actual output.
#
# Usage:
#   scripts/test_all.sh                # default: all tests
#   scripts/test_all.sh --skip-camera  # CI mode (no camera permission)
#   VERBOSE=1 scripts/test_all.sh      # echo every command
#
# Subsequent topic commits (Mac, i.MX, ESP32) append their own checks
# below the foundation sections.

set -u
cd "$(dirname "$0")/.."

SKIP_CAMERA=0
for a in "$@"; do
    case "$a" in
        --skip-camera) SKIP_CAMERA=1 ;;
        --help|-h)
            sed -n '2,15p' "$0"; exit 0 ;;
    esac
done

ARCH="$(uname -m)"
OS="$(uname -s)"
PASS=()
FAIL=()
SKIP=()

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }
hdr()   { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

# Run a labelled test. $1 = label, $2... = command.
run() {
    local label="$1"; shift
    printf '\n→ %-40s ' "$label"
    if [[ "${VERBOSE:-0}" = 1 ]]; then echo; echo "    cmd: $*"; fi
    local out
    if out=$("$@" 2>&1); then
        green "PASS"
        PASS+=("$label")
    else
        red "FAIL"
        echo "$out" | sed 's/^/    | /' | head -20
        FAIL+=("$label")
    fi
}

skip() {
    printf '\n→ %-40s ' "$1"; yellow "SKIP ($2)"
    SKIP+=("$1 ($2)")
}

# ---------------------------------------------------------------------------
hdr "Environment"
echo "host:     $OS / $ARCH"
echo "compiler: $(${CC:-cc} --version | head -1)"
echo "branch:   $(git branch --show-current 2>/dev/null || echo n/a)"
echo "head:     $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

# ---------------------------------------------------------------------------
hdr "Default build (host arch)"
run "make clean"                     bash -c 'make clean >/dev/null 2>&1 || true; true'
run "make (default)"                 make
run "libfacex.a exists"              test -f libfacex.a
run "facex-cli exists"               test -x ./facex-cli
run "libdetect.a exists"             test -f libdetect.a

# ---------------------------------------------------------------------------
hdr "Smoke tests on the default build"
if [[ -f data/edgeface_xs_fp32.bin ]]; then
    run "make test (golden)"         make test
else
    skip "make test (golden)" "data/edgeface_xs_fp32.bin missing"
fi

# ---------------------------------------------------------------------------
hdr "External dependency audit (default build)"
if [[ "$OS" = "Darwin" ]]; then
    run "facex-cli has only system deps" \
        bash -c '! otool -L facex-cli | tail -n +2 | grep -vqE "/usr/lib/libSystem|/System/Library"'
    run "libfacex.a is self-contained" \
        bash -c 'test "$(ar t libfacex.a | grep -cE "\.o$")" -ge 5'
fi

# ---------------------------------------------------------------------------
hdr "Unified bench tool"
if [[ -f data/edgeface_xs_fp32.bin ]]; then
    run "make bench"               make bench
    run "facex-bench --help"       bash -c './facex-bench --help 2>&1 | grep -q "Usage"'
    run "facex-bench md output"    bash -c './facex-bench --iters 5 --warmup 2 --format md   | grep -q "FaceX bench"'
    run "facex-bench csv output"   bash -c './facex-bench --iters 5 --warmup 2 --format csv  | grep -q "label,compiled,active"'
    run "facex-bench json output"  bash -c './facex-bench --iters 5 --warmup 2 --format json | grep -q "stages"'
    run "facex-bench embed-only stage" \
        bash -c './facex-bench --iters 5 --warmup 2 --stage embed --format csv | grep -q ",embed,5,"'
    run "scripts/bench_all.sh produces a sweep table" \
        bash -c 'scripts/bench_all.sh --iters 5 --warmup 2 --configs "default" 2>/dev/null | grep -q "default"'
else
    skip "unified bench" "data/edgeface_xs_fp32.bin missing"
fi

# ---------------------------------------------------------------------------
hdr "Camera benchmark"
if [[ "$OS" = "Darwin" && "$SKIP_CAMERA" = 0 ]]; then
    run "make bench-camera"          make bench-camera
    run "facex-camera-bench --help"  ./facex-camera-bench --help
    # Single bench run, retry once on flake. macOS holds the camera
    # device briefly after a process exits — a 2 s settle gap between
    # back-to-back runs avoids fighting it.
    bench_once() {
        local logf="$1"; shift
        ( "$@" >"$logf" 2>&1 ) &
        local pid=$! waited=0
        # 20 s window — camera stack can need 3-5 s to deliver first frame
        # cold, especially right after a previous bench run released the
        # device. 30 frames at 30 fps then takes ~1 s.
        while kill -0 "$pid" 2>/dev/null; do
            sleep 1; waited=$((waited+1))
            if [[ $waited -gt 20 ]]; then kill "$pid" 2>/dev/null; sleep 1; break; fi
        done
        wait "$pid" 2>/dev/null
        grep -q "fps" "$logf"
    }
    bench_run() {
        local label="$1"; shift
        local logf="$(mktemp -t facex_bench.XXXX)"
        printf '\n→ %-40s ' "$label"
        if bench_once "$logf" "$@"; then
            green "PASS"; PASS+=("$label")
        else
            sleep 2  # let macOS release the device
            if bench_once "$logf" "$@"; then
                green "PASS (retried)"; PASS+=("$label")
            else
                red "FAIL"; sed 's/^/    | /' "$logf" | head -10; FAIL+=("$label")
            fi
        fi
        rm -f "$logf"
    }
    bench_run "camera-only baseline (--no-detect)" ./facex-camera-bench --frames 30 --no-detect
    sleep 2  # gap between back-to-back camera runs
    if [[ -f data/edgeface_xs_fp32.bin ]]; then
        bench_run "full pipeline 30 frames" ./facex-camera-bench --frames 30
    fi
else
    skip "camera bench" "non-Darwin or --skip-camera"
fi

# ---------------------------------------------------------------------------
hdr "Summary"
TOTAL=$(( ${#PASS[@]} + ${#FAIL[@]} + ${#SKIP[@]} ))
green  "passed:  ${#PASS[@]}/$TOTAL"
yellow "skipped: ${#SKIP[@]}/$TOTAL"
if [[ ${#FAIL[@]} -eq 0 ]]; then
    green  "failed:  0/$TOTAL"
    echo
    green  "ALL OK"
    exit 0
else
    red    "failed:  ${#FAIL[@]}/$TOTAL"
    echo
    for f in "${FAIL[@]}"; do red "  ✗ $f"; done
    exit 1
fi
