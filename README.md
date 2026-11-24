# LM_Dec_Long_HybridAlt_W64

HybridAlt_W64 - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (6 seed-repeat runs, see results)**
- Failure class: none

## Provenance

- Source repository: (new) hybrid: GRU + local/global attention stack
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7 + torch.nn

## Differences from the source ([Unified] protocol)

- GRU, then 6 single layers alternating sliding-window-64 RoPE and global NoPE (L G L G L G)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 20.74 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.687 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.397 / test 1.381 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.317 / test 1.307 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 621 MB, inference 218 MB
- Peak GPU memory at length 8192 (batch 1): train 5937 MB, inference 2859 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.28 (keys removed 0.21); 4 pairs @ 2048: 0.26 (keys removed 0.26); 16 pairs @ 512: 0.08 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.06)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.25 (keys removed 0.26); 4 pairs @ 2048: 0.24 (keys removed 0.25); 4 pairs @ 8192: 0.26 (keys removed 0.23); 16 pairs @ 512: 1.00 (keys removed 0.03); 16 pairs @ 2048: 1.00 (keys removed 0.02); 16 pairs @ 8192: 1.00 (keys removed 0.01)
- LM-05 recall `lm05v3s2`: 4 pairs @ 512: 0.27 (keys removed 0.27); 4 pairs @ 2048: 0.27 (keys removed 0.24); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.08 (keys removed 0.07)
- LM-05 recall `lm05v3s3`: 4 pairs @ 512: 0.25 (keys removed 0.24); 4 pairs @ 2048: 0.23 (keys removed 0.26); 16 pairs @ 512: 0.07 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.07)
- LM-05 recall `lm05v3longs2`: 4 pairs @ 512: 1.00 (keys removed 0.05); 4 pairs @ 2048: 1.00 (keys removed 0.03); 4 pairs @ 8192: 1.00 (keys removed 0.04); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.08 (keys removed 0.07); 16 pairs @ 8192: 0.06 (keys removed 0.07)
- LM-06 memory update `lm06`: @512: latest 0.70, unchanged 0.06, stale 0.01; @2048: latest 0.68, unchanged 0.06, stale 0.01
- LM-06 memory update `lm06long`: @512: latest 0.74, unchanged 0.05, stale 0.01; @2048: latest 0.73, unchanged 0.05, stale 0.01; @8192: latest 0.67, unchanged 0.05, stale 0.01
- LM-06 memory update `lm06s2`: @512: latest 0.68, unchanged 0.12, stale 0.05; @2048: latest 0.65, unchanged 0.13, stale 0.04
- LM-06 memory update `lm06s3`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 0.99, unchanged 1.00, stale 0.01
- LM-06 memory update `lm06longs2`: @512: latest 1.00, unchanged 0.99, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00; @8192: latest 1.00, unchanged 1.00, stale 0.00

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check hyb_alt64
python run.py lm02 hyb_alt64      # enwik8 is downloaded to data/ on first use
```
