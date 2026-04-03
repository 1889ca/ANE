// infer_xor_patch.h — Bankai XOR patch support for 1-bit weight modification
// Patches are JSON files that specify rows to flip in weight matrices.
// Format: {"flips": [{"layer": L, "proj": "gate_proj", "row": R}, ...]}
// Applying a flip negates all elements in row R of the specified projection.
// XOR is self-inverse: applying the same patch twice restores original weights.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Map projection names to our weight loading order
// gate_proj = W1 (ffn_gate), up_proj = W3 (ffn_up), down_proj = W2 (ffn_down)
// Also support attention projections for future patches
typedef enum {
    XOR_PROJ_GATE = 0,   // gate_proj / ffn_gate / W1
    XOR_PROJ_UP,         // up_proj / ffn_up / W3
    XOR_PROJ_DOWN,       // down_proj / ffn_down / W2
    XOR_PROJ_Q,          // q_proj / attn_q
    XOR_PROJ_K,          // k_proj / attn_k
    XOR_PROJ_V,          // v_proj / attn_v
    XOR_PROJ_O,          // o_proj / attn_output
    XOR_PROJ_COUNT
} XORProj;

typedef struct {
    int layer;
    XORProj proj;
    int row;
} XORFlip;

typedef struct {
    XORFlip *flips;
    int n_flips;
    char name[128];
} XORPatch;

static XORProj xor_proj_from_str(const char *s) {
    if (strcmp(s, "gate_proj") == 0 || strcmp(s, "ffn_gate") == 0) return XOR_PROJ_GATE;
    if (strcmp(s, "up_proj") == 0 || strcmp(s, "ffn_up") == 0) return XOR_PROJ_UP;
    if (strcmp(s, "down_proj") == 0 || strcmp(s, "ffn_down") == 0) return XOR_PROJ_DOWN;
    if (strcmp(s, "q_proj") == 0 || strcmp(s, "attn_q") == 0) return XOR_PROJ_Q;
    if (strcmp(s, "k_proj") == 0 || strcmp(s, "attn_k") == 0) return XOR_PROJ_K;
    if (strcmp(s, "v_proj") == 0 || strcmp(s, "attn_v") == 0) return XOR_PROJ_V;
    if (strcmp(s, "o_proj") == 0 || strcmp(s, "attn_output") == 0) return XOR_PROJ_O;
    return XOR_PROJ_COUNT;  // unknown
}

// Minimal JSON parser — just enough for the patch format
// Looks for "flips": [...] array with {layer, proj, row} objects
static XORPatch *xor_patch_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open XOR patch: %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = (char *)malloc(sz + 1);
    fread(json, 1, sz, f);
    json[sz] = '\0';
    fclose(f);

    XORPatch *p = (XORPatch *)calloc(1, sizeof(XORPatch));

    // Extract name
    char *name_pos = strstr(json, "\"name\"");
    if (name_pos) {
        char *q1 = strchr(name_pos + 6, '"');
        if (q1) { q1++;
            char *q2 = strchr(q1, '"');
            if (q2) { int len = (int)(q2 - q1); if (len > 127) len = 127;
                memcpy(p->name, q1, len); p->name[len] = '\0'; }
        }
    }

    // Count flips (count occurrences of "layer")
    int count = 0;
    char *scan = json;
    while ((scan = strstr(scan, "\"layer\"")) != NULL) { count++; scan++; }

    p->flips = (XORFlip *)malloc(count * sizeof(XORFlip));
    p->n_flips = 0;

    // Parse each flip: {"layer": N, "proj": "name", "row": N}
    scan = strstr(json, "\"flips\"");
    if (!scan) { free(json); free(p->flips); free(p); return NULL; }

    scan = strchr(scan, '[');
    if (!scan) { free(json); free(p->flips); free(p); return NULL; }

    while (p->n_flips < count) {
        char *obj = strchr(scan, '{');
        if (!obj) break;
        scan = obj + 1;

        int layer = -1, row = -1;
        char proj_str[64] = {0};

        // Find layer
        char *lp = strstr(obj, "\"layer\"");
        if (lp) { lp = strchr(lp + 7, ':'); if (lp) layer = atoi(lp + 1); }

        // Find row
        char *rp = strstr(obj, "\"row\"");
        if (rp) { rp = strchr(rp + 5, ':'); if (rp) row = atoi(rp + 1); }

        // Find proj
        char *pp = strstr(obj, "\"proj\"");
        if (pp) {
            char *q1 = strchr(pp + 6, '"'); if (q1) { q1++;
                char *q2 = strchr(q1, '"');
                if (q2) { int len = (int)(q2 - q1); if (len > 63) len = 63;
                    memcpy(proj_str, q1, len); proj_str[len] = '\0'; }
            }
        }

        if (layer >= 0 && row >= 0 && proj_str[0]) {
            XORProj proj = xor_proj_from_str(proj_str);
            if (proj < XOR_PROJ_COUNT) {
                p->flips[p->n_flips].layer = layer;
                p->flips[p->n_flips].proj = proj;
                p->flips[p->n_flips].row = row;
                p->n_flips++;
            }
        }
    }

    free(json);
    printf("XOR patch: \"%s\" — %d flips loaded from %s\n", p->name, p->n_flips, path);
    return p;
}

// Apply XOR flips to a dequantized fp16 weight matrix [out_ch, in_ch]
// Flipping row r: negate all in_ch elements at row r
static void xor_patch_apply_f16(_Float16 *w, int out_ch, int in_ch,
                                 const XORPatch *patch, int layer, XORProj proj) {
    int applied = 0;
    for (int i = 0; i < patch->n_flips; i++) {
        if (patch->flips[i].layer == layer && patch->flips[i].proj == proj) {
            int row = patch->flips[i].row;
            if (row >= 0 && row < out_ch) {
                _Float16 *r = w + row * in_ch;
                for (int j = 0; j < in_ch; j++)
                    r[j] = -r[j];
                applied++;
            }
        }
    }
    if (applied > 0)
        printf("    XOR: L%d %s — %d rows flipped\n", layer,
               proj == XOR_PROJ_GATE ? "gate" : proj == XOR_PROJ_UP ? "up" :
               proj == XOR_PROJ_DOWN ? "down" : proj == XOR_PROJ_Q ? "q" :
               proj == XOR_PROJ_K ? "k" : proj == XOR_PROJ_V ? "v" : "o",
               applied);
}

static void xor_patch_free(XORPatch *p) {
    if (!p) return;
    free(p->flips);
    free(p);
}
