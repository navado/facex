/*
 * FaceX — Fast face embedding library.
 *
 * 3ms inference, 7MB binary, zero dependencies.
 * Beats ONNX Runtime on CPU with handwritten C + AVX2/AVX-512.
 *
 * Usage:
 *   FaceX* fx = facex_init("edgeface_xs_fp32.bin", NULL);
 *   float embedding[512];
 *   facex_embed(fx, rgb_112x112, embedding);
 *   float sim = facex_similarity(emb1, emb2);
 *   facex_free(fx);
 *
 * https://github.com/facex-engine/facex
 * License: CC BY-NC-SA 4.0
 */

#ifndef FACEX_H
#define FACEX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct FaceX FaceX;

/*
 * Initialize FaceX engine.
 *   weights_path: path to .bin weight file (EFXS or EFXE encrypted)
 *   license_key:  NULL for unencrypted weights, or license string for encrypted
 *   Returns: engine handle, or NULL on error
 */
FaceX* facex_init(const char* weights_path, const char* license_key);

/*
 * Compute 512-dim face embedding.
 *   fx:        engine handle
 *   rgb_hwc:   112x112x3 float32 array, HWC layout, values in [-1, 1]
 *   embedding: output buffer, 512 floats (L2-normalized)
 *   Returns: 0 on success, -1 on error
 */
int facex_embed(FaceX* fx, const float* rgb_hwc, float embedding[512]);

/*
 * Cosine similarity between two embeddings.
 *   Returns: value in [-1, 1], typically >0.3 = same person
 */
float facex_similarity(const float emb1[512], const float emb2[512]);

/*
 * Free engine resources.
 */
void facex_free(FaceX* fx);

/*
 * Get version string.
 */
const char* facex_version(void);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_H */
