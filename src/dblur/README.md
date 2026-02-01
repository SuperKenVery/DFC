```console
CUDA_VISIBLE_DEVICES=1 python -m pdb -c c 1_train_model.py \
  --model SPF_LUT_net \
  --scale 4 \
  --modes sdy \
  --expDir ../models/dblurring-spf \
  --trainDir ../data/GoPro/ \
  --valDir ../data/SRBenchmark/GoPro/ \
  --sample-size 3 \
  --batchSize 8
```
