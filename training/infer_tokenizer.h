// infer_tokenizer.h — BPE tokenizer parsed from GGUF metadata
// Supports Qwen3/GPT2-style BPE with byte-level fallback
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ========== GGUF KV reading helpers ==========

// Read a GGUF string at pos, advance pos. Returns malloc'd string.
static char *tok_gguf_read_str(const uint8_t *data, size_t *pos, size_t fsize) {
    if (*pos + 8 > fsize) return NULL;
    uint64_t len = *(uint64_t *)(data + *pos); *pos += 8;
    if (*pos + len > fsize || len > 1048576) return NULL;
    char *s = (char *)malloc(len + 1);
    memcpy(s, data + *pos, len);
    s[len] = '\0';
    *pos += len;
    return s;
}

// Skip a GGUF value
static void tok_gguf_skip(const uint8_t *data, size_t *pos, uint32_t type, size_t fsize) {
    switch (type) {
        case 0: case 1: case 7: *pos += 1; break;     // u8/i8/bool
        case 2: case 3: *pos += 2; break;              // u16/i16
        case 4: case 5: case 6: *pos += 4; break;      // u32/i32/f32
        case 10: case 11: case 12: *pos += 8; break;   // u64/i64/f64
        case 8: { free(tok_gguf_read_str(data, pos, fsize)); break; }  // string
        case 9: {  // array
            uint32_t et = *(uint32_t *)(data + *pos); *pos += 4;
            uint64_t cnt = *(uint64_t *)(data + *pos); *pos += 8;
            for (uint64_t i = 0; i < cnt; i++) tok_gguf_skip(data, pos, et, fsize);
            break;
        }
    }
}

// Read a GGUF uint32 value
static uint32_t tok_gguf_read_u32(const uint8_t *data, size_t *pos) {
    uint32_t v = *(uint32_t *)(data + *pos); *pos += 4;
    return v;
}

// ========== BPE Tokenizer ==========

// Hash table entry for merge rank lookup
typedef struct BPEEntry {
    char *key;              // "left right" merged string
    int rank;               // lower = higher priority
    struct BPEEntry *next;  // chain for collisions
} BPEEntry;

#define BPE_HASH_SIZE 262144  // power of 2 for fast modulo

typedef struct {
    char **vocab;           // token strings [n_tokens]
    int *token_type;        // token types [n_tokens]
    int n_tokens;
    BPEEntry **hash;        // merge rank hash table
    int n_merges;
    int bos_id, eos_id, unk_id, pad_id;
} BPETokenizer;

static uint32_t bpe_hash(const char *s) {
    uint32_t h = 5381;
    while (*s) h = ((h << 5) + h) ^ (uint8_t)*s++;
    return h & (BPE_HASH_SIZE - 1);
}

static void bpe_hash_insert(BPETokenizer *t, const char *key, int rank) {
    uint32_t h = bpe_hash(key);
    BPEEntry *e = (BPEEntry *)malloc(sizeof(BPEEntry));
    e->key = strdup(key);
    e->rank = rank;
    e->next = t->hash[h];
    t->hash[h] = e;
}

static int bpe_hash_lookup(const BPETokenizer *t, const char *left, const char *right) {
    // Build "left right" key
    size_t ll = strlen(left), rl = strlen(right);
    char *key = (char *)alloca(ll + 1 + rl + 1);
    memcpy(key, left, ll);
    key[ll] = ' ';
    memcpy(key + ll + 1, right, rl);
    key[ll + 1 + rl] = '\0';

    uint32_t h = bpe_hash(key);
    for (BPEEntry *e = t->hash[h]; e; e = e->next)
        if (strcmp(e->key, key) == 0) return e->rank;
    return -1;
}

// Build reverse lookup: token string → token ID
// Uses a simple hash table for O(1) lookup
#define VOCAB_HASH_SIZE 524288

typedef struct VocabEntry {
    char *key;
    int id;
    struct VocabEntry *next;
} VocabEntry;

static VocabEntry **g_vocab_hash = NULL;

static void vocab_hash_build(const BPETokenizer *t) {
    g_vocab_hash = (VocabEntry **)calloc(VOCAB_HASH_SIZE, sizeof(VocabEntry *));
    for (int i = 0; i < t->n_tokens; i++) {
        uint32_t h = bpe_hash(t->vocab[i]) & (VOCAB_HASH_SIZE - 1);
        VocabEntry *e = (VocabEntry *)malloc(sizeof(VocabEntry));
        e->key = t->vocab[i];
        e->id = i;
        e->next = g_vocab_hash[h];
        g_vocab_hash[h] = e;
    }
}

static int vocab_lookup(const char *token) {
    if (!g_vocab_hash) return -1;
    uint32_t h = bpe_hash(token) & (VOCAB_HASH_SIZE - 1);
    for (VocabEntry *e = g_vocab_hash[h]; e; e = e->next)
        if (strcmp(e->key, token) == 0) return e->id;
    return -1;
}

// ========== Parse tokenizer from GGUF ==========

static BPETokenizer *bpe_tokenizer_from_gguf(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open GGUF for tokenizer: %s\n", path); return NULL; }
    struct stat st; fstat(fd, &st);
    size_t fsize = st.st_size;
    const uint8_t *data = (const uint8_t *)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { close(fd); return NULL; }

    // Parse header
    if (fsize < 24 || *(uint32_t *)data != 0x46554747) {
        munmap((void *)data, fsize); close(fd); return NULL;
    }
    uint64_t n_kv = *(uint64_t *)(data + 16);

    BPETokenizer *t = (BPETokenizer *)calloc(1, sizeof(BPETokenizer));
    t->bos_id = -1; t->eos_id = -1; t->unk_id = -1; t->pad_id = -1;
    t->hash = (BPEEntry **)calloc(BPE_HASH_SIZE, sizeof(BPEEntry *));

    // Parse KV pairs, extract tokenizer metadata
    size_t pos = 24;
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = tok_gguf_read_str(data, &pos, fsize);
        if (!key) break;
        uint32_t vtype = *(uint32_t *)(data + pos); pos += 4;

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            uint32_t et = *(uint32_t *)(data + pos); pos += 4;
            uint64_t cnt = *(uint64_t *)(data + pos); pos += 8;
            t->n_tokens = (int)cnt;
            t->vocab = (char **)calloc(cnt, sizeof(char *));
            for (uint64_t j = 0; j < cnt; j++)
                t->vocab[j] = tok_gguf_read_str(data, &pos, fsize);
        } else if (strcmp(key, "tokenizer.ggml.token_type") == 0 && vtype == 9) {
            uint32_t et = *(uint32_t *)(data + pos); pos += 4;
            uint64_t cnt = *(uint64_t *)(data + pos); pos += 8;
            t->token_type = (int *)malloc(cnt * sizeof(int));
            // Elements are i32
            for (uint64_t j = 0; j < cnt; j++) {
                t->token_type[j] = *(int32_t *)(data + pos); pos += 4;
            }
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == 9) {
            uint32_t et = *(uint32_t *)(data + pos); pos += 4;
            uint64_t cnt = *(uint64_t *)(data + pos); pos += 8;
            t->n_merges = (int)cnt;
            for (uint64_t j = 0; j < cnt; j++) {
                char *merge = tok_gguf_read_str(data, &pos, fsize);
                if (merge) {
                    bpe_hash_insert(t, merge, (int)j);
                    free(merge);
                }
            }
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            t->bos_id = (int)tok_gguf_read_u32(data, &pos);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            t->eos_id = (int)tok_gguf_read_u32(data, &pos);
        } else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0) {
            t->unk_id = (int)tok_gguf_read_u32(data, &pos);
        } else if (strcmp(key, "tokenizer.ggml.padding_token_id") == 0) {
            t->pad_id = (int)tok_gguf_read_u32(data, &pos);
        } else {
            tok_gguf_skip(data, &pos, vtype, fsize);
        }
        free(key);
    }

    munmap((void *)data, fsize);
    close(fd);

    if (!t->vocab || t->n_tokens == 0) {
        fprintf(stderr, "No tokenizer vocabulary found in GGUF\n");
        free(t); return NULL;
    }

    // Build vocab reverse lookup
    vocab_hash_build(t);

    printf("Tokenizer: %d tokens, %d merges, bos=%d eos=%d\n",
           t->n_tokens, t->n_merges, t->bos_id, t->eos_id);
    return t;
}

// ========== BPE Encode ==========

// BPE symbol list (doubly-linked for efficient merge)
typedef struct BPESym {
    int start, len;     // byte range in input text
    int prev, next;     // linked list indices (-1 = end)
} BPESym;

static int bpe_encode(const BPETokenizer *t, const char *text, int *out_tokens, int max_tokens) {
    int text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    // Start with one symbol per byte
    int n_sym = text_len;
    BPESym *syms = (BPESym *)malloc((text_len + 1) * sizeof(BPESym));
    char **sym_str = (char **)malloc((text_len + 1) * sizeof(char *));

    for (int i = 0; i < text_len; i++) {
        syms[i].start = i;
        syms[i].len = 1;
        syms[i].prev = i - 1;
        syms[i].next = (i + 1 < text_len) ? i + 1 : -1;
        sym_str[i] = (char *)malloc(2);
        sym_str[i][0] = text[i];
        sym_str[i][1] = '\0';
    }

    // Iterative BPE merge
    int capacity = text_len + 1;
    while (1) {
        int best_rank = INT32_MAX;
        int best_i = -1;

        // Find the merge with lowest rank
        for (int i = 0; i < capacity; i++) {
            if (!sym_str[i] || syms[i].next < 0) continue;
            int j = syms[i].next;
            if (!sym_str[j]) continue;

            int rank = bpe_hash_lookup(t, sym_str[i], sym_str[j]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;  // No more merges

        // Merge best_i and its next
        int j = syms[best_i].next;
        size_t li = strlen(sym_str[best_i]);
        size_t lj = strlen(sym_str[j]);
        char *merged = (char *)malloc(li + lj + 1);
        memcpy(merged, sym_str[best_i], li);
        memcpy(merged + li, sym_str[j], lj);
        merged[li + lj] = '\0';

        free(sym_str[best_i]);
        sym_str[best_i] = merged;
        syms[best_i].len += syms[j].len;

        // Remove j from linked list
        free(sym_str[j]);
        sym_str[j] = NULL;
        syms[best_i].next = syms[j].next;
        if (syms[j].next >= 0)
            syms[syms[j].next].prev = best_i;
    }

    // Collect surviving symbols, look up token IDs
    int n_out = 0;
    for (int i = 0; i >= 0 && n_out < max_tokens;) {
        if (!sym_str[i]) { i++; continue; }
        int id = vocab_lookup(sym_str[i]);
        if (id >= 0) {
            out_tokens[n_out++] = id;
        } else {
            // Byte-level fallback: encode each byte as <0xNN> token
            for (int b = 0; b < (int)strlen(sym_str[i]) && n_out < max_tokens; b++) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (uint8_t)sym_str[i][b]);
                int byte_id = vocab_lookup(byte_tok);
                if (byte_id >= 0) out_tokens[n_out++] = byte_id;
            }
        }
        i = syms[i].next;
        if (i < 0) break;
    }

    // Cleanup
    for (int i = 0; i < capacity; i++) free(sym_str[i]);
    free(sym_str);
    free(syms);

    return n_out;
}

// ========== BPE Decode ==========

static const char *bpe_decode_token(const BPETokenizer *t, int id) {
    if (id >= 0 && id < t->n_tokens) return t->vocab[id];
    return "";
}

static void bpe_tokenizer_free(BPETokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->n_tokens; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->token_type);
    for (int i = 0; i < BPE_HASH_SIZE; i++) {
        BPEEntry *e = t->hash[i];
        while (e) { BPEEntry *n = e->next; free(e->key); free(e); e = n; }
    }
    free(t->hash);
    if (g_vocab_hash) {
        for (int i = 0; i < VOCAB_HASH_SIZE; i++) {
            VocabEntry *e = g_vocab_hash[i];
            while (e) { VocabEntry *n = e->next; free(e); e = n; }
        }
        free(g_vocab_hash);
        g_vocab_hash = NULL;
    }
    free(t);
}
