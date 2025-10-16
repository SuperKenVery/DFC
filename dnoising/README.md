```console
CUDA_VISIBLE_DEVICES=2 python -m pdb -c c 1_train_model.py \
  --model SPF_LUT_net \
  --scale 4 \
  --modes sdy \
  --expDir ../models/dnoising-spf \
  --trainDir ../data/DIV2K \
  --valDir ../data/SRBenchmark/ \
  --sample-size 3 \
  --batchSize 8
```
