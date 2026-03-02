#!/usr/bin/env python3
"""Create a small fine-tuning corpus of simple English stories (style-similar to TinyStories).
Tests whether near-distribution fine-tuning works where cross-domain fails."""

import struct
import numpy as np
from sentencepiece import SentencePieceProcessor

OUTPUT = "simple_stories.bin"
TOKENIZER = "/tmp/llama2c/tokenizer.model"

# Simple, TinyStories-adjacent stories — modern, simple vocabulary, child-like
STORIES = [
    "Once upon a time, there was a brave little robot named Bolt. Bolt loved to help people in the village. Every morning, Bolt would wake up and check if anyone needed help. One day, a little girl named Luna lost her cat. Bolt searched everywhere and found the cat hiding under a bridge. Luna was so happy she gave Bolt a big hug.",
    "There was a magical garden where flowers could talk. The roses would sing songs and the daisies would tell jokes. A little boy named Sam discovered the garden one sunny afternoon. He sat among the flowers and listened to their stories. The sunflower told him about the time it grew taller than the fence. Sam laughed and visited the garden every day after school.",
    "A tiny mouse named Pip lived inside a clock tower. Every hour, Pip would wind the clock to keep it ticking. The townspeople never knew that a mouse kept their clock running. One winter, Pip got sick and could not wind the clock. The clock stopped and the whole town was confused. A kind girl named Ella found Pip and nursed him back to health. Pip wound the clock again and the town cheered.",
    "In a forest far away, there lived a fox who loved to paint. Every day, the fox would paint pictures of the trees and the sky. The other animals would come to watch. The bear liked the paintings of mountains. The rabbit liked the paintings of meadows. One day, the fox painted a rainbow and all the animals agreed it was the most beautiful painting ever.",
    "A little fish named Coral lived in a deep blue pond. Coral dreamed of seeing the ocean. Her mother told her the ocean was very far away. One day, a friendly turtle offered to take Coral on a journey. They swam through rivers and streams. Finally, they reached the ocean. Coral was amazed by how big and beautiful it was. She thanked the turtle and swam home to tell her friends.",
    "There was a cloud named Fluffy who was afraid of thunder. Every time a storm came, Fluffy would hide behind the tallest mountain. The sun told Fluffy that thunder was just noise and could not hurt anyone. Fluffy was still scared. One day, Fluffy was brave and stayed in the sky during a storm. The thunder boomed but Fluffy was fine. After that, Fluffy was never afraid again.",
    "A penguin named Pete wanted to learn how to fly. All the other birds could fly but Pete could only waddle and swim. Pete tried jumping off rocks and flapping his wings very fast. He could not fly. A wise owl told Pete that swimming was just like flying underwater. Pete realized the owl was right. He dove into the ocean and felt like he was flying through the water.",
    "In a small town, there was a baker who made the best bread. People came from far away to buy his bread. The baker had a secret ingredient: he always added a little bit of honey. One day, the baker ran out of honey. He was very worried. A kind beekeeper heard about his problem and brought him a whole jar of honey. The baker was grateful and gave the beekeeper free bread for a year.",
    "A caterpillar named Cal was waiting to become a butterfly. Cal ate leaves and grew bigger every day. The other bugs told Cal that becoming a butterfly was the most wonderful thing. Cal was excited but also nervous. One day, Cal built a cocoon and fell asleep inside. When Cal woke up, he had beautiful wings. Cal flew up into the sky and felt the warm sunshine on his new wings.",
    "There was a little star who shone very brightly. The other stars were jealous because the little star was the brightest in the sky. The moon told the other stars that each of them was special in their own way. Some stars were big, some were small, some twinkled fast and some twinkled slow. The other stars realized the moon was right and they all shone together happily.",
] * 100  # Repeat 100x for sufficient tokens

def main():
    sp = SentencePieceProcessor(model_file=TOKENIZER)
    bos_id = sp.bos_id()

    all_tokens = []
    for story in STORIES:
        tokens = [bos_id] + sp.encode(story)
        all_tokens.extend(tokens)

    arr = np.array(all_tokens, dtype=np.uint16)
    print(f"Tokens: {len(arr):,} ({len(arr)*2/1e6:.1f} MB)")
    print(f"Stories: {len(STORIES)}")

    with open(OUTPUT, "wb") as f:
        f.write(arr.tobytes())

    print(f"\nFirst 50 tokens decoded:")
    print(repr(sp.decode(arr[:50].tolist())))

if __name__ == "__main__":
    main()
