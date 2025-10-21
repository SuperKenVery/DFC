import sys
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, "../sr")
from discriminator import Discriminator

# Load real image
real_img_path = "/data2/xuyuheng/DFCs/gan-ft/data/DIV2K/HR/0001.png"
real_img = Image.open(real_img_path).convert('RGB')

# Transform to tensor
transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

real_tensor = transform(real_img).unsqueeze(0)  # Add batch dimension
print(f"Real image shape: {real_tensor.shape}")

# Create fake image (just noise)
fake_tensor = torch.randn_like(real_tensor)

device = torch.device('cuda')
real_tensor = real_tensor.to(device)
fake_tensor = fake_tensor.to(device)

# Create discriminator
model_D = Discriminator(device=device)
model_D = model_D.to(device)

# Optimizer
opt_D = optim.Adam(model_D.parameters(), lr=0.01)

print("\n" + "="*60)
print("Testing Discriminator Training")
print("="*60)

# Train for a few steps
for step in range(200):
    opt_D.zero_grad()

    # Get losses and scores
    loss_real, score_real = model_D(real_tensor, for_real=True)
    loss_fake, score_fake = model_D(fake_tensor, for_real=False)

    total_loss = (loss_real + loss_fake).mean()

    total_loss.backward()
    opt_D.step()

    # loss_fake=score
    # loss_real=alpha*log(sigmoid(score)) + (1-alpha)*log(1-sigmoid(score))

    if step % 5 == 0:
        with torch.no_grad():
            # Check accuracy using raw scores
            correct = (score_real > score_fake).float().mean().item() * 100
            advantage = (score_real - score_fake).mean().item()

        print(f"\nStep {step}:")
        print(f"  Loss Real: {loss_real.mean().item():.4f}")
        print(f"  Loss Fake: {loss_fake.mean().item():.4f}")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"  Score Real: {score_real.mean().item():.4f}")
        print(f"  Score Fake: {score_fake.mean().item():.4f}")
        print(f"  Accuracy: {correct:.1f}%")
        print(f"  Advantage: {advantage:.4f}")
