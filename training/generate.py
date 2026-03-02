#!/usr/bin/env python3
"""Generate text from ANE-trained stories110M checkpoint vs pretrained baseline.

Forward pass matches the ANE training kernels exactly:
  - Channel-first [DIM, T] layout
  - Causal masking with RoPE positional encoding
  - RMSNorm, multi-head SDPA, SiLU FFN
"""

import math, struct, sys, os, time
import numpy as np

DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS = 768, 2048, 12, 256, 32000, 12
HD = DIM // HEADS
CKPT_PATH = 'ane_stories110M_ckpt.bin'
MODEL_PATH = '../../assets/models/stories110M.bin'
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', '..', 'assets', 'models', 'tokenizer.bin')


class Tokenizer:
    def __init__(self, path):
        self.vocab = []
        with open(path, 'rb') as f:
            struct.unpack('i', f.read(4))  # max_len
            for _ in range(VOCAB):
                struct.unpack('f', f.read(4))  # score
                slen = struct.unpack('i', f.read(4))[0]
                self.vocab.append(f.read(slen).decode('utf-8', errors='replace'))

    def decode(self, token_id):
        if 0 <= token_id < len(self.vocab):
            s = self.vocab[token_id]
            if s.startswith('<0x') and s.endswith('>'):
                try: return chr(int(s[3:-1], 16))
                except: return s
            return s
        return ''


def load_pretrained(path):
    """Load weights from llama2.c format (stories110M.bin)."""
    W = {}
    with open(path, 'rb') as f:
        cfg = struct.unpack('7i', f.read(28))
        dim, hidden, nlayers, heads, _, vocab_size, seq_len = cfg
        V = abs(vocab_size)
        shared = vocab_size > 0

        embed = np.frombuffer(f.read(V * dim * 4), np.float32).reshape(V, dim).copy()
        W['embed'] = embed
        for L in range(nlayers):
            W[f'rms1_{L}'] = np.frombuffer(f.read(dim * 4), np.float32).copy()
        for L in range(nlayers):
            W[f'Wq{L}'] = np.frombuffer(f.read(dim * dim * 4), np.float32).reshape(dim, dim).copy()
        for L in range(nlayers):
            W[f'Wk{L}'] = np.frombuffer(f.read(dim * dim * 4), np.float32).reshape(dim, dim).copy()
        for L in range(nlayers):
            W[f'Wv{L}'] = np.frombuffer(f.read(dim * dim * 4), np.float32).reshape(dim, dim).copy()
        for L in range(nlayers):
            W[f'Wo{L}'] = np.frombuffer(f.read(dim * dim * 4), np.float32).reshape(dim, dim).copy()
        for L in range(nlayers):
            W[f'rms2_{L}'] = np.frombuffer(f.read(dim * 4), np.float32).copy()
        for L in range(nlayers):
            W[f'W1_{L}'] = np.frombuffer(f.read(hidden * dim * 4), np.float32).reshape(hidden, dim).copy()
        for L in range(nlayers):
            W[f'W2_{L}'] = np.frombuffer(f.read(dim * hidden * 4), np.float32).reshape(dim, hidden).copy()
        for L in range(nlayers):
            W[f'W3_{L}'] = np.frombuffer(f.read(hidden * dim * 4), np.float32).reshape(hidden, dim).copy()
        W['rms_final'] = np.frombuffer(f.read(dim * 4), np.float32).copy()
    return W


def load_checkpoint(path):
    """Load weights from ANE training checkpoint."""
    W = {}
    with open(path, 'rb') as f:
        hdr = f.read(96)
        step = struct.unpack_from('i', hdr, 8)[0]
        loss = struct.unpack_from('f', hdr, 44)[0]

        wq_sz, wo_sz = DIM * DIM, DIM * DIM
        w1_sz, w2_sz, w3_sz = HIDDEN * DIM, DIM * HIDDEN, HIDDEN * DIM
        adam_skip = (wq_sz*2 + wq_sz*2 + wq_sz*2 + wo_sz*2 +
                     w1_sz*2 + w2_sz*2 + w3_sz*2 + DIM*2 + DIM*2) * 4

        for L in range(NLAYERS):
            W[f'Wq{L}'] = np.frombuffer(f.read(wq_sz*4), np.float32).reshape(DIM, DIM).copy()
            W[f'Wk{L}'] = np.frombuffer(f.read(wq_sz*4), np.float32).reshape(DIM, DIM).copy()
            W[f'Wv{L}'] = np.frombuffer(f.read(wq_sz*4), np.float32).reshape(DIM, DIM).copy()
            W[f'Wo{L}'] = np.frombuffer(f.read(wo_sz*4), np.float32).reshape(DIM, DIM).copy()
            W[f'W1_{L}'] = np.frombuffer(f.read(w1_sz*4), np.float32).reshape(HIDDEN, DIM).copy()
            W[f'W2_{L}'] = np.frombuffer(f.read(w2_sz*4), np.float32).reshape(DIM, HIDDEN).copy()
            W[f'W3_{L}'] = np.frombuffer(f.read(w3_sz*4), np.float32).reshape(HIDDEN, DIM).copy()
            W[f'rms1_{L}'] = np.frombuffer(f.read(DIM*4), np.float32).copy()
            W[f'rms2_{L}'] = np.frombuffer(f.read(DIM*4), np.float32).copy()
            f.seek(adam_skip, 1)

        W['rms_final'] = np.frombuffer(f.read(DIM*4), np.float32).copy()
        f.seek(DIM * 2 * 4, 1)  # skip adam
        W['embed'] = np.frombuffer(f.read(VOCAB*DIM*4), np.float32).reshape(VOCAB, DIM).copy()

    return W, step, loss


def rmsnorm(x, w):
    """x: [DIM, T], w: [DIM]. Returns [DIM, T]."""
    ss = np.mean(x * x, axis=0, keepdims=True) + 1e-5
    return x * (1.0 / np.sqrt(ss)) * w[:, None]


def apply_rope(Q, K, T):
    """Apply RoPE to Q, K in [HEADS, T, HD] layout."""
    for h in range(HEADS):
        for i in range(0, HD, 2):
            freq = 1.0 / (10000.0 ** (i / HD))
            for t in range(T):
                cos_v = math.cos(t * freq)
                sin_v = math.sin(t * freq)
                q0, q1 = Q[h, t, i], Q[h, t, i + 1]
                Q[h, t, i] = q0 * cos_v - q1 * sin_v
                Q[h, t, i + 1] = q0 * sin_v + q1 * cos_v
                k0, k1 = K[h, t, i], K[h, t, i + 1]
                K[h, t, i] = k0 * cos_v - k1 * sin_v
                K[h, t, i + 1] = k0 * sin_v + k1 * cos_v
    return Q, K


def forward(W, tokens):
    """Full forward pass matching ANE training kernels. Returns logits at last position."""
    T = len(tokens)
    x = W['embed'][tokens].T.astype(np.float32).copy()  # [DIM, T]

    # Causal mask [T, T]
    mask = np.full((T, T), -65504.0, dtype=np.float32)
    for t1 in range(T):
        for t2 in range(t1 + 1):
            mask[t1, t2] = 0.0

    for L in range(NLAYERS):
        xn = rmsnorm(x, W[f'rms1_{L}'])
        Q = (W[f'Wq{L}'] @ xn).reshape(HEADS, HD, T).transpose(0, 2, 1)
        K = (W[f'Wk{L}'] @ xn).reshape(HEADS, HD, T).transpose(0, 2, 1)
        V = (W[f'Wv{L}'] @ xn).reshape(HEADS, HD, T).transpose(0, 2, 1)
        Q, K = apply_rope(Q, K, T)

        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(HD) + mask[None, :, :]
        scores -= np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=-1, keepdims=True)

        attn = (probs @ V).transpose(0, 2, 1).reshape(DIM, T)
        x2 = x + W[f'Wo{L}'] @ attn

        x2n = rmsnorm(x2, W[f'rms2_{L}'])
        h1 = W[f'W1_{L}'] @ x2n
        h3 = W[f'W3_{L}'] @ x2n
        x = x2 + W[f'W2_{L}'] @ (h1 * (1.0 / (1.0 + np.exp(-h1))) * h3)

    x = rmsnorm(x, W['rms_final'])
    return W['embed'] @ x[:, -1]  # [VOCAB]


def generate(W, tokenizer, max_tokens=128, temperature=0.8):
    """Autoregressive generation."""
    tokens = [1]  # BOS

    for _ in range(max_tokens):
        if len(tokens) >= SEQ:
            break
        logits = forward(W, tokens)

        if temperature < 0.01:
            next_tok = int(np.argmax(logits))
        else:
            logits = logits / temperature
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)
            probs = np.clip(probs, 0, None)
            probs /= probs.sum()
            next_tok = int(np.random.choice(VOCAB, p=probs))

        if next_tok == 2:  # EOS
            break
        tokens.append(next_tok)
        print(tokenizer.decode(next_tok), end='', flush=True)

    print()
    return len(tokens) - 1


if __name__ == '__main__':
    tokenizer = Tokenizer(TOKENIZER_PATH)
    print(f"Tokenizer: {len(tokenizer.vocab)} tokens\n")

    # Load pretrained baseline
    print("=" * 60)
    print("PRETRAINED (no training, with RoPE)")
    print("=" * 60)
    W_pre = load_pretrained(MODEL_PATH)

    np.random.seed(42)
    for i in range(3):
        t0 = time.time()
        n = generate(W_pre, tokenizer, max_tokens=64, temperature=0.8)
        print(f"  [{time.time()-t0:.1f}s, {n} tokens]")
        print("-" * 60)

    print("\nPretrained greedy:")
    n = generate(W_pre, tokenizer, max_tokens=64, temperature=0.0)
    print(f"  [{n} tokens]")
    print()

    # Load trained checkpoint
    if not os.path.exists(CKPT_PATH):
        print(f"No checkpoint at {CKPT_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("TRAINED (with RoPE)")
    print("=" * 60)
    W_train, step, loss = load_checkpoint(CKPT_PATH)
    print(f"Checkpoint: step={step}, loss={loss:.4f}\n")

    np.random.seed(42)
    for i in range(3):
        t0 = time.time()
        n = generate(W_train, tokenizer, max_tokens=64, temperature=0.8)
        print(f"  [{time.time()-t0:.1f}s, {n} tokens]")
        print("-" * 60)

    print("\nTrained greedy:")
    n = generate(W_train, tokenizer, max_tokens=64, temperature=0.0)
    print(f"  [{n} tokens]")
