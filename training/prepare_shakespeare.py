#!/usr/bin/env python3
"""Tokenize Shakespeare (Complete Works) for ANE fine-tuning.
Produces shakespeare.bin (flat uint16 token IDs, Llama 2 BPE 32K vocab)."""

import struct
import numpy as np
from sentencepiece import SentencePieceProcessor

INPUT = "/tmp/shakespeare_complete.txt"
OUTPUT = "shakespeare.bin"
TOKENIZER = "/tmp/llama2c/tokenizer.model"

# Gutenberg markers
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

def main():
    sp = SentencePieceProcessor(model_file=TOKENIZER)
    bos_id = sp.bos_id()

    with open(INPUT, "r", encoding="utf-8") as f:
        text = f.read()

    # Strip Gutenberg header/footer
    start = text.find(START_MARKER)
    if start >= 0:
        text = text[text.index("\n", start) + 1:]
    end = text.find(END_MARKER)
    if end >= 0:
        text = text[:end]

    text = text.strip()
    print(f"Text: {len(text):,} chars, {len(text.split()):,} words")

    # Split into chunks on blank lines (scenes/sections)
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    print(f"Chunks: {len(chunks):,}")

    # Tokenize each chunk with BOS prefix
    all_tokens = []
    for chunk in chunks:
        tokens = [bos_id] + sp.encode(chunk)
        all_tokens.extend(tokens)

    arr = np.array(all_tokens, dtype=np.uint16)
    print(f"Tokens: {len(arr):,} ({len(arr)*2/1e6:.1f} MB)")

    with open(OUTPUT, "wb") as f:
        f.write(arr.tobytes())

    # Sanity: decode first 50 tokens
    print(f"\nFirst 50 tokens decoded:")
    first = arr[:50].tolist()
    print(repr(sp.decode(first)))

if __name__ == "__main__":
    main()
