# \[CVPR 2024\] Look-Up Table Compression for Efficient Image Restoration

Yinglong Li, [Jiacheng Li](https://ddlee-cn.github.io/), [Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong/)

[CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Look-Up_Table_Compression_for_Efficient_Image_Restoration_CVPR_2024_paper.html)

![Overview of DFC](https://github.com/leenas233/DFC/blob/main/docs/DFC_overview.png)

## Quick Start

The project uses TOML configuration files for all settings. No CLI arguments needed beyond the experiment directory.

```bash
# 1. Create a new experiment with config file
python -m src.new_experiment models/my_experiment

# 2. Edit models/my_experiment/config.toml to customize settings

# 3. Run training
accelerate launch run.py src.sr.1_train_model -e models/my_experiment

# 4. Export LUT
accelerate launch run.py src.sr.2_compress_lut_from_net -e models/my_experiment

# 5. Finetune LUT
accelerate launch run.py src.sr.3_finetune_compress_lut -e models/my_experiment
```

## Configuration File

The `config.toml` file contains all experiment settings with documentation:

```toml
[model]
model = "SPF_LUT_net"    # Model architecture
scale = 4                 # Upscaling factor
branches = 3              # Number of parallel SR-LUT branches per stage
stages = 2               # Number of stages

[data]
train_dir = "data/DIV2K"
val_dir = "data/SRBenchmark"
batch_size = 32

[train]
total_iter = 200000
lr0 = 0.001
lr1 = 0.0001

[export_lut]
checkpoint_iter = 200000  # Which checkpoint to export

[export_lut.dfc]
enabled = false           # Enable DFC compression
diagonal_width = 2
sampling_interval = 5

[finetune_lut]
export_lut_iter = 200000  # Which exported LUT to finetune
total_iter = 200000
lr0 = 0.0001
```

## Dataset

| task             | training dataset                                      | testing dataset                                                                                                                               |
| ---------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| super-resolution | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)    | Set5, Set14, [BSDS100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), Urban100, [Manga109](http://www.manga109.org/en/)   |
| denoising        | DIV2K                                                 | Set12, BSD68                                                                                                                                  |
| deblocking       | DIV2K                                                 | [Classic5](https://github.com/cszn/DnCNN/tree/master/testsets/classic5), [LIVE1](https://live.ece.utexas.edu/research/quality/subjective.htm) |
| deblurring       | [GoPro](https://seungjunnah.github.io/Datasets/gopro) | GoPro test set                                                                                                                                |

## Pretrained Models

Some pretrained LUTs and their compressed version can be download [here](https://drive.google.com/drive/folders/1nxPzhpLdZut-16T_Z3b-5Oo9uU4Dbe1h?usp=drive_link).

## Multi-GPU Training

```bash
accelerate launch --multi_gpu --num_processes=4 run.py src.sr.1_train_model -e models/my_experiment
```

## Contact

If you have any questions, feel free to contact me any time by e-mail `yllee@mail.ustc.edu.cn`

## Citation

If you found our implementation useful, please consider citing our paper:

```bibtex
@InProceedings{Li_2024_CVPR,
	author = {Li, Yinglong and Li, Jiacheng and Xiong, Zhiwei},
	title = {Look-Up Table Compression for Efficient Image Restoration},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June}, year = {2024}, pages = {26016-26025}
}
```

## Acknowledgement

This work is based on the following works, thank the authors a lot.

[SR-LUT](https://github.com/yhjo09/SR-LUT)

[MuLUT](https://github.com/ddlee-cn/MuLUT/tree/main)
