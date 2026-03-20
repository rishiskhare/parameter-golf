# Recurrent Depth Sharing + Ternary Codebook Quantization

**Mean val_bpb: TBD** (pending benchmark run)

## Key Techniques

1. **Recurrent depth sharing**: Prelude / shared core / coda block architecture — reuses transformer layers for extra effective depth without proportional parameter cost.

2. **Native ternary linear layers**: Straight-through estimation with ternary codebook weights. Compresses extremely well under zlib.

3. **Adaptive recurrence at eval**: Extra recurrent steps at test time for improved predictions without retraining.

4. **Test-time training (TTT)**: Optional per-document fine-tuning at evaluation for domain adaptation.

5. **FP16 embedding export**: `tok_emb` and `embed_proj` kept in float16 to avoid quantization error compounding through both input and output paths.

6. **Muon decoupled weight decay** (0.02): Improves generalization and quantization robustness for matrix params.

7. **Sliding window evaluation** (stride=64, seq_len=2048): Dense context scoring for every token.

8. **Longer training context** (seq_len=2048): Better long-range pattern learning.

9. **Extended warmdown** (2500 iters): Longer cooldown for better final weights.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| TBD  | TBD      | TBD     | TBD   | TBD     |

## Run Command

```bash
RUN_ID=recurrent_ternary_v1 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log
```
