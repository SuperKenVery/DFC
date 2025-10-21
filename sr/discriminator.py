# Idea from https://github.com/XPixelGroup/HYPIR

import open_clip
from open_clip.factory import CLIP
import torch
from torch import nn
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from vision_aided_loss.cv_losses import multilevel_loss
from PIL import Image
from accelerate import Accelerator, logging, DeepSpeedPlugin
from typing import List

class Discriminator(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, device: torch.device = "cpu"):
        super().__init__()

        self.clip, _, _preprocess = open_clip.create_model_and_transforms(
            "convnext_xxlarge",
            pretrained="laion2b_s34b_b82k_augreg_soup",
            device=device
        )
        self.mld = MultiLevelD(in_channels=[768, 1536, 3072, 1024])
        # TODO: Use multileve_hinge_loss which is WGAN style.
        self.loss_fn = multilevel_loss(alpha=0.8)

        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32), persistent=False)

    def forward(self, x, for_real=True, for_G=False):
        x = (x - self.image_mean[:, None, None]) / self.image_std[:, None, None]
        with torch.no_grad(): # Don't train convnext
            backbone_features = self.convnext_get_feats(x)
        multilevel_features = self.mld(backbone_features)

        # Compute loss
        loss = self.loss_fn(multilevel_features, for_real=for_real, for_G=for_G)

        # Compute raw scores (average across all levels)
        total_score = 0
        count = 0
        for each in multilevel_features:
            # Flatten all dimensions except batch, then average
            score = each.view(each.size(0), -1).mean(dim=1)
            total_score += score
            count += 1
        avg_score = total_score / count

        return loss, avg_score

    def convnext_get_feats(self, x) -> List[torch.Tensor]:
        x, intermediates = self.clip.visual.trunk.forward_intermediates(x)
        x = self.clip.visual.trunk.forward_head(x)
        x = self.clip.visual.head(x)
        return intermediates[1:] + [x]

    def print_shapes(self, name: str, tensors: List[torch.Tensor]):
        print(f"Shapes of {name} (len={len(tensors)}):")
        for x in tensors:
            print(f"\t{x.shape}")


# https://github.com/nupurkmr9/vision-aided-gan/blob/95fc55beefad3e868783beab421108c5baf583aa/vision_aided_loss/cv_discriminator.py#L11-L45
class MultiLevelD(nn.Module):
    def __init__(self, in_channels: List[int], out_ch=1, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True), down=1):
        super().__init__()

        self.decoder = nn.ModuleList()
        self.in_channels = in_channels
        for in_ch in in_channels[:-1]:
            self.decoder.append(nn.Sequential(
                                BlurPool(in_ch, pad_type='zero', stride=1, pad_off=1) if down > 1 else nn.Identity(),
                                spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2 if down > 1 else 1, padding=1 if down == 1 else 0)),
                                activation,
                                BlurPool(out_ch, pad_type='zero', stride=1),
                                spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)))
                                )
        self.decoder.append(nn.Sequential(spectral_norm(nn.Linear(in_channels[-1], out_ch)), activation))
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)

    def forward(self, x, c=None):
        assert len(self.decoder)==len(self.in_channels)
        final_pred = []
        for decoder, part_x in zip(self.decoder[:-1], x[:-1]):
            final_pred.append(decoder(part_x).squeeze(1))

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
    image_tensor = vF.pil_to_tensor(image).float().unsqueeze(0)[:, :, :100, :100]
    print("Image tensor shape", image_tensor.shape)

    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            zero_stage=3,
            # offload_param_device="cpu",
            hf_ds_config={
                "train_micro_batch_size_per_gpu": 1,
                "zero_optimization": {
                    "stage": 3,
                }
            }
        )
    )

    # with torch.autocast(dtype=torch.float16, device_type="cuda"):
    dis = Discriminator().to("cuda")
    dis = accelerator.prepare(dis)
    image_tensor = image_tensor.to(accelerator.device)
    realness = dis(image_tensor)
    print(realness)
