/*
 * FaceX Detect — engine implementation.
 *
 * This is the Sprint-1 scaffold: data structures, public-API wiring,
 * threshold setters, and a do-nothing `detect_run` that returns 0 faces.
 * Subsequent sprints fill in the backbone, FPN, heads, anchor decoding,
 * and NMS.
 */

#include "detect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DETECT_VERSION_STR "0.1.0-scaffold"

#define DETECT_DEFAULT_SCORE_THRESHOLD 0.5f
#define DETECT_DEFAULT_NMS_THRESHOLD   0.4f

struct Detect {
    /* Tunables (settable via API) */
    float score_threshold;
    float nms_threshold;

    /* Weight blob ownership */
    uint8_t* weights;
    size_t   weights_len;

    /* Reserved for future sprints: layer table, scale tables,
     * activation buffers, anchor tables, NMS scratch, ... */
};

static int load_file(const char* path, uint8_t** out, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return -1; }
    rewind(f);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); return -1; }
    *out = buf;
    *out_len = (size_t)sz;
    return 0;
}

Detect* detect_init(const char* weights_path) {
    if (!weights_path) return NULL;

    Detect* det = (Detect*)calloc(1, sizeof(Detect));
    if (!det) return NULL;

    det->score_threshold = DETECT_DEFAULT_SCORE_THRESHOLD;
    det->nms_threshold   = DETECT_DEFAULT_NMS_THRESHOLD;

    if (load_file(weights_path, &det->weights, &det->weights_len) != 0) {
        free(det);
        return NULL;
    }

    /* TODO(sprint 3): parse header, set up layer table.
     * For now, retain the blob so detect_free() unloads it. */
    return det;
}

int detect_run(Detect* det,
               const uint8_t* rgb_hwc,
               int width, int height,
               DetectFace* out, int max_faces) {
    if (!det || !rgb_hwc || !out || max_faces < 0) return -1;
    if (width <= 0 || height <= 0) return -1;

    /* TODO(sprint 8+): full forward pass.
     * Scaffold returns zero detections so callers can integrate now. */
    (void)det; (void)rgb_hwc; (void)width; (void)height; (void)out; (void)max_faces;
    return 0;
}

void detect_free(Detect* det) {
    if (!det) return;
    free(det->weights);
    free(det);
}

const char* detect_version(void) {
    return DETECT_VERSION_STR;
}

void detect_set_score_threshold(Detect* det, float threshold) {
    if (!det) return;
    if (threshold < 0.0f) threshold = 0.0f;
    if (threshold > 1.0f) threshold = 1.0f;
    det->score_threshold = threshold;
}

void detect_set_nms_threshold(Detect* det, float threshold) {
    if (!det) return;
    if (threshold < 0.0f) threshold = 0.0f;
    if (threshold > 1.0f) threshold = 1.0f;
    det->nms_threshold = threshold;
}
