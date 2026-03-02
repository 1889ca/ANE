# ane-finetune: On-Device Fine-Tuning via Apple Neural Engine

Design exploration for a CLI tool that fine-tunes language models entirely on-device using the ANE.

## Value Proposition

**"Fine-tune a language model on your data. It never leaves your Mac."**

- Privacy: training data stays on-device. No cloud API, no upload, no trust required.
- GPU-free: trains on ANE, leaving GPU available for display and other apps. Train while you work.
- Simple: `ane-finetune --model base.bin --data my_corpus.txt --out finetuned.bin`

## What It Is Not

- Not a competitor to MLX/PyTorch for training large models. Those are better tools for that.
- Not fast. An H100 will always win on throughput.
- Not general-purpose ML training. This is specifically "take a pretrained LLM, adapt it to your data."

## MVP Scope

### Supported Models

Start with one model family where we control the full stack:

**TinyLlama architecture** (Llama 2 family, RoPE, RMSNorm, SwiGLU):
- 110M (current, proven — useful for validation)
- 350M (stretch goal — the smallest "actually useful" size)

Same architecture, different dimensions. The MIL kernel generators already parameterize on dim/hidden/heads.

### User Interface

```bash
# Minimal: fine-tune on a text file
ane-finetune --data my_notes.txt

# Full options
ane-finetune \
  --model stories110M.bin \        # pretrained weights (llama2c format)
  --tokenizer tokenizer.model \    # sentencepiece BPE model
  --data corpus.txt \              # plain text, one doc per line (or paragraph)
  --out finetuned.bin \            # output checkpoint
  --steps 1000 \                   # training steps
  --lr 3e-4 \                      # learning rate
  --seq 256 \                      # sequence length
  --accum 50                       # gradient accumulation steps

# Generate from fine-tuned model
ane-generate --model finetuned.bin --prompt "Once upon a time"
```

### Data Pipeline

1. User provides plain text file(s)
2. Tool tokenizes using sentencepiece (bundled tokenizer matching the base model)
3. Random sampling of seq-length windows during training (current approach)
4. No batching in MVP (batch=1 with gradient accumulation)

### Output

A checkpoint file containing:
- Fine-tuned weights
- Adam optimizer state (for resuming)
- Training metadata (steps, loss curve, config)

Plus a simple inference path: `ane-generate` that loads the checkpoint and does autoregressive generation on ANE.

## What Needs to Change From Current Code

### Already done (training/ as-is)
- ANE forward/backward pass
- Weight loading from llama2c format
- Checkpoint save/load with Adam state
- Cosine LR schedule
- Text generation (generate.py)
- Gradient accumulation

### Must do
1. **Decouple model dimensions from compile-time constants.** Current: `#define DIM 768` everywhere. Needed: runtime Config struct. This is the biggest refactor — touches struct definitions, allocation, and kernel generation. But the MIL generators already take dim/hidden/heads as parameters to the generation functions, so the kernel side is partially ready.

2. **CLI for data/model/tokenizer paths.** Current: hardcoded paths. Easy fix, mostly done (--steps, --accum already exist).

3. **Tokenization step.** Current: separate `prepare_data.py` script. Needed: either bundle tokenization into the C binary (link sentencepiece C API) or keep as a preprocessing step with clear docs.

4. **Bundled base model + tokenizer.** The tool needs to ship with (or download) a pretrained base model. For 110M that's ~440MB. For 350M, ~1.4GB.

### Nice to have
- Progress bar with ETA and loss curve
- Validation loss on held-out split (auto 90/10)
- Early stopping
- LoRA / parameter-efficient fine-tuning (dramatically reduces compilation overhead since fewer weights change)
- Export to GGUF or CoreML format for use in other tools

## Verification Plan

The core question: **does fine-tuning on domain-specific data make the model measurably better at that domain?**

### Test 1: Style Transfer (Qualitative)

Fine-tune 110M on a specific author's writing (e.g., public domain — Shakespeare, Austen, Poe). Generate from fine-tuned model and compare to base model outputs.

**Expected:** Fine-tuned model adopts vocabulary and sentence structure of the target author. Base model produces generic TinyStories output.

**How to run:**
```bash
# Prepare corpus
cat shakespeare_sonnets.txt | ane-finetune --steps 2000 --data - --out shakespeare.bin

# Compare generations
ane-generate --model stories110M.bin --prompt "The king spoke"    # base
ane-generate --model shakespeare.bin --prompt "The king spoke"    # finetuned
```

### Test 2: Domain Perplexity (Quantitative)

Split domain corpus 90/10. Fine-tune on 90%. Measure perplexity on held-out 10%. Compare to base model perplexity on same held-out set.

**Expected:** Fine-tuned perplexity significantly lower than base model on domain data. Base model perplexity on general data should increase somewhat (catastrophic forgetting is expected and fine for narrow use).

**How to run:**
```bash
# Auto-split and evaluate
ane-finetune --data recipes.txt --eval-split 0.1 --steps 2000 --out recipes.bin
# Output: base_ppl=142.3  finetuned_ppl=28.7  on held-out 10%
```

### Test 3: Convergence Sanity

Training loss should decrease monotonically (smoothed). If loss plateaus early or diverges, something is wrong with the training loop or hyperparameters.

**Already verified:** Current training loop shows clean loss curves on TinyStories (4.31 → 3.61 over 5000 steps). This test validates that domain-specific data behaves similarly.

### Test 4: Memorization Check

Generate from the fine-tuned model and check that outputs are not verbatim copies of training data. For a 110M model with limited capacity, some memorization is expected on small corpora, but the model should be generating novel combinations.

### Test 5: Round-Trip Correctness

Save checkpoint, reload, continue training. Loss should be continuous (no jump at reload). Already implemented and tested.

## Experimental Results (March 2026)

### Test: Near-distribution fine-tuning (simple stories)
- Corpus: 1000 simple English stories, TinyStories-adjacent style, 92K tokens
- LR=1e-4, accum=50, grad clipping max_norm=1.0
- **Loss: 2.02 → 0.65 in 500 steps** (10 weight updates)
- Gradient norms: 4.4 → 2.4 (naturally decreasing, healthy)
- Training loop works correctly

### Test: Cross-domain fine-tuning (Shakespeare)
- Corpus: Complete Works of Shakespeare, 1.7M tokens
- **FAILED at every learning rate tested:**
  - LR=1e-4: loss 6.9 → 20+ (diverges by step 200)
  - LR=5e-5 + warmup: loss 6.9 → 20+ (diverges by step 300)
  - LR=1e-5: loss flat at ~7.0 (no learning in 500 steps)
  - LR=1e-4 + freeze 10/12 layers: loss 6.9 → 15-16 (slow degradation)
- Gradient norms explode: 10 → 600+ without clipping, stabilize at 7-10 with layer freezing
- Root cause: fp16 backward pass through 12 transformer layers compounds precision errors on OOD data

### Diagnosis

The fp16-only compute path is the fundamental limitation. With the entire forward/backward on ANE in fp16:
- In-distribution data → small loss → small gradients → stable training
- Cross-domain data → high loss (~7 vs ~4 in-distribution) → larger gradients → fp16 precision compounds through 12 backward layers → gradient corruption → catastrophic divergence

Standard mitigations (mixed-precision backward in fp32, loss scaling) are not available because the ANE only supports fp16 compute. Gradient clipping helps symptoms but not root cause — the gradients reaching CPU are already corrupted from fp16 backward propagation.

### Implication for Product

The ane-finetune value proposition shifts from "fine-tune on any data" to "**adapt within a domain**":
- Fine-tune a children's story model on YOUR children's stories → works
- Fine-tune on private notes in similar style → likely works
- Fine-tune a story model to write Shakespeare → does not work
- Cross-domain transfer (stories → code, stories → technical writing) → likely fails

This narrows the product scope but doesn't kill it. The privacy story still holds for in-domain adaptation. But the base model must already be trained on similar data to the fine-tuning target.

## Honest Risks

1. **fp16 backward limits fine-tuning to near-distribution data.** (CONFIRMED) Cross-domain fine-tuning fails due to gradient precision loss. The model must already have reasonable loss on the target data (< ~5 CE) for stable training.

2. **110M might be too small to be useful.** TinyStories was specifically designed for tiny models. Real-world domains (code, technical writing, conversation) may require 350M+ for the fine-tuned model to produce coherent, useful output. We won't know until we test.

3. **Compilation overhead makes short runs expensive.** The first batch pays ~6 seconds of compilation. For a 1000-step fine-tune at 74ms/step, that's 6s compile + 74s train = 80s total. Compilation is 7.5% overhead. Acceptable, but it means "fine-tune for 10 steps to test" is dominated by compilation.

4. **No LoRA.** Full fine-tuning updates all parameters, which means full recompilation every accumulation cycle. LoRA would freeze most weights and only update low-rank adapters, dramatically reducing compilation. But LoRA requires architectural changes to the MIL kernels.

## Phased Approach

### Phase 1: Validate the value prop (current model, minimal CLI)
- Add --data, --model, --tokenizer CLI args to train_large.m
- Bundle tokenization (or document preprocessing step clearly)
- Run Tests 1-3 with a real domain corpus (not TinyStories)
- **Ship as blog post / HN discussion piece with code**

### Phase 2: Make it usable
- Runtime model dimensions (Config struct, dynamic allocation)
- Progress output (loss curve, ETA)
- Validation split + perplexity reporting
- ane-generate as standalone binary
- Bundle a pretrained 110M model + tokenizer

### Phase 3: Make it useful
- Scale to 350M (validate ANE handles larger models)
- Batching (batch=4, 3-4x throughput)
- LoRA (reduce compilation overhead)
- Export to standard formats
