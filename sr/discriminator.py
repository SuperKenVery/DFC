# Idea from https://github.com/XPixelGroup/HYPIR

import open_clip
from open_clip.factory import CLIP
import torch
from torch import nn
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from vision_aided_loss.cv_losses import multilevel_loss
from PIL import Image
from accelerate import Accelerator, logging, DeepSpeedPlugin
import tensor_parallel as tp

class Discriminator(nn.Module):
    def __init__(self, device: torch.device = "cpu"):
        super().__init__()

        self.clip, _, _preprocess = open_clip.create_model_and_transforms(
            "convnext_xxlarge",
            pretrained="laion2b_s34b_b82k_augreg_soup",
            device=device
        )
        self.mld = MultiLevelD()
        self.loss_fn = multilevel_loss(alpha=0.8)

    def forward(self, x):
        backbone_features = self.convnext_get_feats(x)
        multilevel_features = self.mld(backbone_features)
        return self.loss_fn(multilevel_features, for_real=True)

    def convnext_get_feats(self, x):
        x, intermediates = self.clip.visual.trunk.forward_intermediates(x)
        x = self.clip.visual.trunk.forward_head(x)
        x = self.clip.visual.head(x)
        return intermediates[1:] + [x]


# https://github.com/nupurkmr9/vision-aided-gan/blob/95fc55beefad3e868783beab421108c5baf583aa/vision_aided_loss/cv_discriminator.py#L11-L45
class MultiLevelD(nn.Module):
    def __init__(self, level=1, in_ch1=1, in_ch2=1, out_ch=1, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True), down=1):
        super().__init__()

        self.decoder = nn.ModuleList()
        self.level = level
        for _ in range(level-1):
            self.decoder.append(nn.Sequential(
                                BlurPool(in_ch1, pad_type='zero', stride=1, pad_off=1) if down > 1 else nn.Identity(),
                                spectral_norm(nn.Conv2d(in_ch1, out_ch, kernel_size=3, stride=2 if down > 1 else 1, padding=1 if down == 1 else 0)),
                                activation,
                                BlurPool(out_ch, pad_type='zero', stride=1),
                                spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)))
                                )
        self.decoder.append(nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation))
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)

    def forward(self, x, c=None):

        final_pred = []
        for i in range(self.level-1):
            final_pred.append(self.decoder[i](x[i]).squeeze(1))

        h = self.decoder[-1](x[-1].float())
        out = self.out(h)

        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)

        final_pred.append(out)
        # final_pred = torch.cat(final_pred, 1)
        return final_pred

if __name__=="__main__":
    from torchvision.transforms import functional as vF
    image_path = "/data/xyh/DFCs/gan-ft/data/DIV2K/HR/0001.png"
    image = Image.open(image_path)
    image_tensor = vF.pil_to_tensor(image).float().unsqueeze(0)
    print("Image tensor shape", image_tensor.shape)

    # accelerator = Accelerator(
    #     deepspeed_plugin=DeepSpeedPlugin(
    #         zero_stage=3,
    #         offload_param_device="cpu",
    #         hf_ds_config={
    #             "train_micro_batch_size_per_gpu": 1,
    #             "zero_optimization": {
    #                 "stage": 3,
    #             }
    #         }
    #     )
    # )

    with torch.autocast(dtype=torch.float16, device_type="cuda"):
        dis = Discriminator().to("cuda")
        realness = dis(image_tensor.to("cuda"))
    print(realness)
