# ANE Training Optimization Findings

Research log for optimizing Stories110M training on Apple Neural Engine.
Each entry documents what was tried, what worked, what didn't, and why.

Read `optimization-log.md` for the full chronological record.

## Quick Reference

| Optimization | Result | File |
|---|---|---|
| Classifier kernel on ANE | cls 16ms -> 2.5ms | `optimization-log.md#1` |
| Batched vvexpf for xent | **Regression** - don't do this | `optimization-log.md#3` |
| Parallel dW queues | 218ms -> 149ms/step (1.47x) | `optimization-log.md#4` |

## Current Bottleneck Profile (50 steps, accum=100)

```
148.7 ms/step total
  ane=13.2  io=7.2  cls=2.7  rms_fwd=0.1
  elem=60.2 [xent=24.1 memcpy=13.6 rms_bwd=8.8 resid=3.7 embed=0.9 embed_bwd=9.1]
```

Top remaining targets:
1. **xent=24ms** - cross-entropy softmax, already cache-optimal on CPU
2. **memcpy=14ms** - dW capture copies (malloc+memcpy per layer per step)
3. **embed_bwd=9ms** - embed dW wait + scatter-add
4. **rms_bwd=9ms** - 25 rmsnorm_bwd calls per step
