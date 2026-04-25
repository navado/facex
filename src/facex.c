/*
 * facex.c — Public API implementation for FaceX library.
 * See include/facex.h for documentation.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/facex.h"

#define FACEX_VERSION "1.0.0"

/* Include the engine (main() excluded via FACEX_LIB) */
#ifndef FACEX_LIB
#define FACEX_LIB
#endif
#include "edgeface_engine.c"

struct FaceX {
    Weights weights;
    int ready;
};

FaceX* facex_init(const char* weights_path, const char* license_key) {
    if (license_key) g_license_key = license_key;

    FaceX* fx = (FaceX*)calloc(1, sizeof(FaceX));
    if (!fx) return NULL;

    if (engine_init(weights_path, &fx->weights) != 0) {
        free(fx);
        return NULL;
    }

    fx->ready = 1;
    return fx;
}

int facex_embed(FaceX* fx, const float* rgb_hwc, float embedding[512]) {
    if (!fx || !fx->ready) return -1;

    /* HWC -> CHW conversion */
    float input_chw[3 * 112 * 112];
    for (int h = 0; h < 112; h++)
        for (int w = 0; w < 112; w++)
            for (int c = 0; c < 3; c++)
                input_chw[c * 112 * 112 + h * 112 + w] = rgb_hwc[(h * 112 + w) * 3 + c];

    edgeface_forward(input_chw, &fx->weights, embedding);
    return 0;
}

float facex_similarity(const float emb1[512], const float emb2[512]) {
    float dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < 512; i++) {
        dot += emb1[i] * emb2[i];
        n1 += emb1[i] * emb1[i];
        n2 += emb2[i] * emb2[i];
    }
    float denom = sqrtf(n1) * sqrtf(n2);
    return denom > 1e-8f ? dot / denom : 0.0f;
}

void facex_free(FaceX* fx) {
    if (!fx) return;
    if (fx->weights.raw) free(fx->weights.raw);
    if (fx->weights.tensors) free(fx->weights.tensors);
    free(fx);
}

const char* facex_version(void) {
    return FACEX_VERSION;
}
