#!/usr/bin/env python
# coding: utf-8

# # Compressing and Representing Information
# 
# This notebook is a supplement to the Compressing and Representing Information Chapter of the [Hands-On Generative AI with Transformers and Diffusion Models](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/) book. This notebooks includes:
# 
# * The code from the book
# * Additional examples
# * Exercise solutions

# ## AutoEncoders
# 
# ### Preparing the Data

# In[1]:


from datasets import load_dataset

mnist = load_dataset("ylecun/mnist")
mnist


# In[2]:


mnist["train"]["image"][1]


# In[ ]:





# In[8]:


# from genaibook.core import show_images

show_images(mnist["train"]["image"][:4])


# In[9]:


import matplotlib as mpl

mpl.rcParams["image.cmap"] = "gray_r"


# In[59]:


import matplotlib.pyplot as plt
import math
import numpy as np
import torch # 导入torch以支持Tensor输入

# --- show_images 函数定义 ---
def show_images(images, titles=None, ncols=None, nrows=None, imsize=3, figsize=None, suptitle=None): # 新增 suptitle 参数
    """
    在Jupyter中显示图像列表或一个包含多张图像的PyTorch张量。

    参数:
    images (list or torch.Tensor): 
        - 如果是列表：包含图像数据（例如PIL Image对象、NumPy数组或PyTorch张量）的列表。
        - 如果是PyTorch张量：期望形状为 (B, C, H, W) 或 (B, H, W)，其中 B 是批次大小。
          函数会将其视为多张图像，并自动拆分。
    titles (list, optional): 每张图像的标题列表。如果提供，长度应与实际显示的图像数量相同。
    ncols (int, optional): 每行的列数。如果指定，将优先使用。
    nrows (int, optional): 每行的行数。如果指定，将优先使用。
                           特别适用于 'batch_vs_preds' 这种明确分行的情况。
                           如果同时指定了 ncols 和 nrows，那么 nrows 会优先用于计算布局。
    imsize (float, optional): 单张图像在 Matplotlib 图形中的尺寸（英寸）。
                              默认为 3.0。如果提供了 figsize，此参数会被忽略。
    figsize (tuple, optional): Matplotlib图形的总大小 (宽度, 高度)。
                               如果指定，imsize 参数将被忽略。
    suptitle (str, optional): 整个图形的标题。
    """
    
    # 1. 统一图像数据格式为列表
    if isinstance(images, torch.Tensor):
        if images.ndim == 2: # (H, W) -> assuming a single grayscale image
            image_list = [images]
        elif images.ndim == 3: # (C, H, W) or (B, H, W) for grayscale batch
            if images.shape[0] in [1, 3, 4]: # Assume (C, H, W)
                # If it's a single image with channel dim, treat as one image
                image_list = [images]
            else: # Assume (B, H, W)
                image_list = [img for img in images]
        elif images.ndim == 4: # (B, C, H, W)
            image_list = [img for img in images]
        else:
            raise ValueError(f"Unsupported torch.Tensor dimensions: {images.ndim}")
    else:
        # Assume it's already a list of image-like objects
        image_list = images

    n_images = len(image_list)
    if n_images == 0:
        print("没有图像可以显示。")
        return

    # 2. 确定布局 (行数和列数)
    if nrows is not None:
        nrows = int(nrows) # 确保是整数
        ncols = math.ceil(n_images / nrows)
    elif ncols is not None:
        ncols = int(ncols) # 确保是整数
        nrows = math.ceil(n_images / ncols)
    else:
        # 尝试创建一个接近正方形的布局
        ncols = math.ceil(math.sqrt(n_images))
        nrows = math.ceil(n_images / ncols)

    # 3. 计算 figsize
    if figsize is None:
        # imsize 控制每张子图的大小
        figsize = (ncols * imsize, nrows * imsize)
    
    # 4. 创建图形和子图
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # 确保axes是一个扁平的1D数组，方便迭代
    if n_images == 1:
        axes = np.array([axes]) # 如果只有一张图，subplots返回的是单个Axes对象，需要包裹成数组
    else:
        axes = axes.flatten() # 否则，如果不是1D数组，就扁平化

    # 5. 遍历图像并绘制
    for i, img_data in enumerate(image_list):
        if i >= len(axes): 
            break # 理论上不会发生，但以防万一

        ax = axes[i]

        img_to_show = None
        # 处理 PyTorch Tensor
        if isinstance(img_data, torch.Tensor):
            # 核心修改：添加 .detach()
            img_to_show = img_data.detach().cpu().numpy()
            # PyTorch通常是 (C, H, W) 或 (H, W)
            if img_to_show.ndim == 3 and img_to_show.shape[0] in [1, 3, 4]:
                # 如果是 (C, H, W)，转为 (H, W, C)
                img_to_show = np.transpose(img_to_show, (1, 2, 0))
            # 移除单通道维度，例如 (H, W, 1) -> (H, W)
            if img_to_show.ndim == 3 and img_to_show.shape[-1] == 1:
                img_to_show = img_to_show.squeeze(-1)
            # 如果是浮点数图像，确保其值在 [0, 1] 范围内
            # 这对于模型输出尤其重要，因为它们可能不是严格的0-1
            if img_to_show.dtype == np.float32 or img_to_show.dtype == np.float64:
                img_to_show = np.clip(img_to_show, 0.0, 1.0)
        # 处理 PIL Image
        elif hasattr(img_data, 'convert'):
            img_to_show = np.array(img_data)
        # 处理 NumPy 数组 (或默认情况)
        else:
            img_to_show = img_data

        # 尝试检测并应用灰度 colormap
        if img_to_show.ndim == 2 or (img_to_show.ndim == 3 and img_to_show.shape[-1] == 1):
            ax.imshow(img_to_show, cmap='gray')
        else:
            ax.imshow(img_to_show)

        ax.axis('off') # 关闭坐标轴，让图像更干净

        if titles and i < len(titles):
            ax.set_title(titles[i])
        elif titles is None:
            ax.set_title(f"Image {i+1}")

    # 6. 隐藏未使用的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    if suptitle: # 添加整个图形的标题
        fig.suptitle(suptitle)
    plt.tight_layout() # 调整子图参数，使之填充整个图像区域，避免重叠
    plt.show()


# In[10]:


show_images(mnist["train"]["image"][:4])


# In[11]:


import torch

torch.manual_seed(1337);


# In[12]:


from torchvision import transforms


def mnist_to_tensor(samples):
    t = transforms.ToTensor()
    samples["image"] = [t(image) for image in samples["image"]]
    return samples


# In[13]:


mnist = mnist.with_transform(mnist_to_tensor)
mnist["train"] = mnist["train"].shuffle(seed=1337)


# In[14]:


x = mnist["train"]["image"][0]
x.min(), x.max()


# In[15]:


show_images(mnist["train"]["image"][0])


# In[16]:


from torch.utils.data import DataLoader

bs = 64
train_dataloader = DataLoader(mnist["train"]["image"], batch_size=bs)


# ### Modeling the Encoder

# In[17]:


from torch import nn


def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


# In[18]:


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512)
        self.conv4 = conv_block(512, 1024)
        self.linear = nn.Linear(1024, 16)

    def forward(self, x):
        x = self.conv1(x)  # (batch size, 128, 14, 14)
        x = self.conv2(x)  # (bs, 256, 7, 7)
        x = self.conv3(x)  # (bs, 512, 3, 3)
        x = self.conv4(x)  # (bs, 1024, 1, 1)
        # Keep batch dimension when flattening
        x = self.linear(x.flatten(start_dim=1))  # (bs, 16)
        return x


# In[19]:


mnist["train"]["image"][0].shape


# In[20]:


in_channels = 1

x = mnist["train"]["image"][0][None, :]
encoder = Encoder(in_channels).eval()

encoded = encoder(x)
encoded.shape


# In[21]:


encoded


# In[22]:


batch = next(iter(train_dataloader))
encoded = Encoder(in_channels=1)(batch)
batch.shape, encoded.shape


# ### Modeling the Decoder

# In[23]:


def conv_transpose_block(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=0,
    with_act=True,
):
    modules = [
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
    ]
    if with_act:  # Controling this will be handy later
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


# In[24]:


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.linear = nn.Linear(
            16, 1024 * 4 * 4
        )  # note it's reshaped in forward
        self.t_conv1 = conv_transpose_block(1024, 512)
        self.t_conv2 = conv_transpose_block(512, 256, output_padding=1)
        self.t_conv3 = conv_transpose_block(256, out_channels, output_padding=1)

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)  # (bs, 1024*4*4)
        x = x.reshape((bs, 1024, 4, 4))  # (bs, 1024, 4, 4)
        x = self.t_conv1(x)  # (bs, 512, 7, 7)
        x = self.t_conv2(x)  # (bs, 256, 14, 14)
        x = self.t_conv3(x)  # (bs, 1, 28, 28)
        return x


# In[25]:


decoded_batch = Decoder(x.shape[0])(encoded)
decoded_batch.shape


# ### Training

# In[26]:


class AutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


# In[27]:


model = AutoEncoder(1)


# In[29]:


import torchsummary

torchsummary.summary(model, input_size=(1, 28, 28), device="cpu")


# In[32]:


def get_device():
	import torch
	# 1. 确定最佳可用设备
	if torch.cuda.is_available():
	    device = "cuda"
	elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
	    # MPS (Metal Performance Shaders) is for Apple Silicon GPUs
	    device = "mps"
	else:
	    device = "cpu"
	print(f"Using device: {device}")
	return device


# In[33]:


import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm.notebook import tqdm, trange

# from genaibook.core import get_device

num_epochs = 10
lr = 1e-4

device = get_device()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

losses = []  # List to store the loss values for plotting
for _ in (progress := trange(num_epochs, desc="Training")):
    for _, batch in (
        inner := tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    ):
        batch = batch.to(device)

        # Pass through the model and obtain reconstructed images
        preds = model(batch)

        # Compare the prediction with the original images
        loss = F.mse_loss(preds, batch)

        # Display loss and store for plotting
        inner.set_postfix(loss=f"{loss.cpu().item():.3f}")
        losses.append(loss.item())

        # Update the model parameters with the optimizer based on this loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    progress.set_postfix(loss=f"{loss.cpu().item():.3f}", lr=f"{lr:.0e}")


# In[34]:


plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("AutoEncoder – Training Loss Curve")
plt.show()


# In[35]:


eval_bs = 16
eval_dataloader = DataLoader(mnist["test"]["image"], batch_size=eval_bs)


# In[36]:


model.eval()
with torch.inference_mode():
    eval_batch = next(iter(eval_dataloader))
    predicted = model(eval_batch.to(device)).cpu()


# In[40]:


batch_vs_preds = torch.cat((eval_batch, predicted))
show_images(batch_vs_preds, imsize=1, nrows=2)


# ### Exploring the Latent Space
# 

# In[41]:


def plot_activation_fn(fn, name):
    x = torch.linspace(-5, 5, 100)
    y = fn(x)
    plt.plot(x, y, label=name)
    plt.legend()


plt.title("Activation Functions")
plot_activation_fn(F.relu, "ReLU")
plot_activation_fn(F.sigmoid, "Sigmoid")


# In[42]:


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(in_channels, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024),
        )
        self.linear = nn.Linear(1024, latent_dims)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv_layers(x)
        x = self.linear(x.reshape(bs, -1))
        return x


# In[43]:


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dims):
        super().__init__()

        self.linear = nn.Linear(latent_dims, 1024 * 4 * 4)
        self.t_conv_layers = nn.Sequential(
            conv_transpose_block(1024, 512),
            conv_transpose_block(512, 256, output_padding=1),
            conv_transpose_block(
                256, out_channels, output_padding=1, with_act=False
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)
        x = x.reshape((bs, 1024, 4, 4))
        x = self.t_conv_layers(x)
        x = self.sigmoid(x)
        return x


# In[44]:


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


# In[45]:


def train(model, num_epochs=10, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

    model.train()  # Put model in training mode
    losses = []
    for _ in (progress := trange(num_epochs, desc="Training")):
        for _, batch in (
            inner := tqdm(
                enumerate(train_dataloader), total=len(train_dataloader)
            )
        ):
            batch = batch.to(device)

            # Pass through the model and obtain another set of images
            preds = model(batch)

            # Compare the prediction with the original images
            loss = F.mse_loss(preds, batch)

            # Display loss and store for plotting
            inner.set_postfix(loss=f"{loss.cpu().item():.3f}")
            losses.append(loss.item())

            # Update the model parameters with the optimizer based on this loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        progress.set_postfix(loss=f"{loss.cpu().item():.3f}", lr=f"{lr:.0e}")
    return losses


# In[46]:


ae_model = AutoEncoder(in_channels=1, latent_dims=2)
ae_model.to(device)


# In[47]:


losses = train(ae_model)


# In[48]:


plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve (two latent dimensions)")
plt.show()


# In[49]:


ae_model.eval()
with torch.inference_mode():
    eval_batch = next(iter(eval_dataloader))
    predicted = ae_model(eval_batch.to(device)).cpu()


# In[50]:


batch_vs_preds = torch.cat((eval_batch, predicted))
show_images(batch_vs_preds, imsize=1, nrows=2)


# ### Visualizing the Latent Space
# 

# In[51]:


images_labels_dataloader = DataLoader(mnist["test"], batch_size=512)


# In[52]:


import pandas as pd

df = pd.DataFrame(
    {
        "x": [],
        "y": [],
        "label": [],
    }
)

for batch in tqdm(
    iter(images_labels_dataloader), total=len(images_labels_dataloader)
):
    encoded = ae_model.encode(batch["image"].to(device)).cpu()
    new_items = {
        "x": [t.item() for t in encoded[:, 0]],
        "y": [t.item() for t in encoded[:, 1]],
        "label": batch["label"],
    }
    df = pd.concat([df, pd.DataFrame(new_items)], ignore_index=True)


# In[53]:


plt.figure(figsize=(10, 8))

for label in range(10):
    points = df[df["label"] == label]
    plt.scatter(points["x"], points["y"], label=label, marker=".")

plt.legend();


# In[54]:


N = 16  # We'll generate 16 points
z = torch.rand((N, 2)) * 8 - 4


# In[55]:


plt.figure(figsize=(10, 8))

for label in range(10):
    points = df[df["label"] == label]
    plt.scatter(points["x"], points["y"], label=label, marker=".")

plt.scatter(z[:, 0], z[:, 1], label="z", marker="s", color="black")
plt.legend();


# In[60]:


ae_decoded = ae_model.decode(z.to(device))
show_images(ae_decoded.cpu(), imsize=1, nrows=1, suptitle="AutoEncoder")


# ## Variational AutoEncoders (VAEs)

# ### VAE Encoders and Decoders
# 

# In[61]:


class VAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(in_channels, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024),
        )

        # Define fully connected layers for mean and log-variance
        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv_layers(x)
        x = x.reshape(bs, -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return (mu, logvar)


# In[62]:


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x):
        # Returns mu, log_var
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Obtain parameters of the normal (Gaussian) distribution
        mu, logvar = self.encode(x)

        # Sample from the distribution
        std = torch.exp(0.5 * logvar)
        z = self.sample(mu, std)

        # Decode the latent point to pixel space
        reconstructed = self.decode(z)

        # Return the reconstructed image, and also the mu and logvar
        # so we can compute a distribution loss
        return reconstructed, mu, logvar

    def sample(self, mu, std):
        # Reparametrization trick
        # Sample from N(0, I), translate and scale
        eps = torch.randn_like(std)
        return mu + eps * std


# ### Training the VAE
# 

# In[63]:


def vae_loss(batch, reconstructed, mu, logvar):
    bs = batch.shape[0]

    # Reconstruction loss from the pixels - 1 per image
    reconstruction_loss = F.mse_loss(
        reconstructed.reshape(bs, -1),
        batch.reshape(bs, -1),
        reduction="none",
    ).sum(dim=-1)

    # KL-divergence loss, per input image
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    # Combine both losses and get the mean across images
    loss = (reconstruction_loss + kl_loss).mean(dim=0)

    return (loss, reconstruction_loss, kl_loss)


# In[64]:


def train_vae(model, num_epochs=10, lr=1e-4):
    model = model.to(device)
    losses = {
        "loss": [],
        "reconstruction_loss": [],
        "kl_loss": [],
    }

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)
    for _ in (progress := trange(num_epochs, desc="Training")):
        for _, batch in (
            inner := tqdm(
                enumerate(train_dataloader), total=len(train_dataloader)
            )
        ):
            batch = batch.to(device)

            # Pass through the model
            reconstructed, mu, logvar = model(batch)

            # Compute the losses
            loss, reconstruction_loss, kl_loss = vae_loss(
                batch, reconstructed, mu, logvar
            )

            # Display loss and store for plotting
            inner.set_postfix(loss=f"{loss.cpu().item():.3f}")
            losses["loss"].append(loss.item())
            losses["reconstruction_loss"].append(
                reconstruction_loss.mean().item()
            )
            losses["kl_loss"].append(kl_loss.mean().item())

            # Update model parameters based on the total loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        progress.set_postfix(loss=f"{loss.cpu().item():.3f}", lr=f"{lr:.0e}")
    return losses


# In[65]:


vae_model = VAE(in_channels=1, latent_dims=2)


# In[66]:


losses = train_vae(vae_model, num_epochs=10, lr=1e-4)


# In[67]:


for k, v in losses.items():
    plt.plot(v, label=k)
plt.legend();


# In[68]:


plt.plot(losses["loss"])
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("VAE - Training Loss Curve")
plt.show()


# In[69]:


plt.plot(losses["kl_loss"])
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("VAE – KL Loss Component")
plt.show()


# In[70]:


plt.plot(losses["reconstruction_loss"])
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("VAE – Reconstruction Loss Component")
plt.show()


# In[71]:


vae_model.eval()
with torch.inference_mode():
    eval_batch = next(iter(eval_dataloader))
    predicted, mu, logvar = (v.cpu() for v in vae_model(eval_batch.to(device)))


# In[72]:


batch_vs_preds = torch.cat((eval_batch, predicted))
show_images(batch_vs_preds, imsize=1, nrows=2)


# In[73]:


df = pd.DataFrame(
    {
        "x": [],
        "y": [],
        "label": [],
    }
)

for batch in tqdm(
    iter(images_labels_dataloader), total=len(images_labels_dataloader)
):
    mu, _ = vae_model.encode(batch["image"].to(device))
    mu = mu.to("cpu")
    new_items = {
        "x": [t.item() for t in mu[:, 0]],
        "y": [t.item() for t in mu[:, 1]],
        "label": batch["label"],
    }
    df = pd.concat([df, pd.DataFrame(new_items)], ignore_index=True)


# In[74]:


plt.figure(figsize=(10, 8))

for label in range(10):
    points = df[df["label"] == label]
    plt.scatter(points["x"], points["y"], label=label, marker=".")

plt.legend();


# In[75]:


z = torch.normal(0, 1, size=(10, 2))
ae_decoded = ae_model.decode(z.to(device))
vae_decoded = vae_model.decode(z.to(device))


# In[76]:


show_images(ae_decoded.cpu(), imsize=1, nrows=1, suptitle="AutoEncoder")
show_images(vae_decoded.cpu(), imsize=1, nrows=1, suptitle="VAE")


# In[77]:


plt.figure(figsize=(10, 8))

for label in range(10):
    points = df[df["label"] == label]
    plt.scatter(points["x"], points["y"], label=label, marker=".")

plt.vlines(-0.8, ymin=-4, ymax=4, linestyle="dashed", colors="black")
plt.legend();


# In[78]:


import numpy as np

with torch.inference_mode():
    inputs = []
    for y in np.linspace(-2, 2, 10):
        inputs.append([-0.8, y])
    z = torch.tensor(inputs, dtype=torch.float32).to(device)
    decoded = vae_model.decode(z)
show_images(decoded.cpu(), imsize=1, nrows=1)
     


# In[79]:


inputs = []
for x in np.linspace(-2, 2, 20):
    for y in np.linspace(-2, 2, 20):
        inputs.append([x, y])
z = torch.tensor(inputs, dtype=torch.float32).to(device)
decoded = vae_model.to(device).decode(z)


# In[80]:


show_images(decoded.cpu(), imsize=0.4, nrows=20)


# ## CLIP

# ### Using CLIP, step by step
# 

# In[83]:


import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# from genaibook.core import SampleURL

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = 'https://hellorfimg.zcool.cn/large/1103065079.jpg'
image = Image.open(requests.get(url, stream=True).raw)


# In[84]:


image_inputs = processor(images=image, return_tensors="pt")
pixel_values = image_inputs["pixel_values"]
pixel_values.shape, pixel_values.min(), pixel_values.max()


# In[85]:


processor.image_processor


# In[86]:


width, height = image.size
crop_length = min(image.size)

left = (width - crop_length) / 2
top = (height - crop_length) / 2
right = (width + crop_length) / 2
bottom = (height + crop_length) / 2

cropped = image.crop((left, top, right, bottom))
cropped


# In[87]:


with torch.inference_mode():
    output = clip.vision_model(pixel_values.to(device))
image_embeddings = output.pooler_output
image_embeddings.shape


# In[88]:


prompts = [
    "a photo of a lion",
    "a photo of a zebra",
]

# Padding makes sure all inputs have the same length
text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
text_inputs


# In[89]:


text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

with torch.inference_mode():
    text_output = clip.text_model(**text_inputs)


# In[90]:


text_embeddings = text_output.pooler_output
text_embeddings.shape


# In[91]:


print(clip.text_projection)
print(clip.visual_projection)


# In[92]:


with torch.inference_mode():
    text_embeddings = clip.text_projection(text_embeddings)
    image_embeddings = clip.visual_projection(image_embeddings)
text_embeddings.shape, image_embeddings.shape


# In[93]:


text_embeddings = text_embeddings / text_embeddings.norm(
    p=2, dim=-1, keepdim=True
)
image_embeddings = image_embeddings / image_embeddings.norm(
    p=2, dim=-1, keepdim=True
)


# In[94]:


similarities = torch.matmul(text_embeddings, image_embeddings.T)
similarities


# In[95]:


similarities = 100 * torch.matmul(text_embeddings, image_embeddings.T)
similarities.softmax(dim=0).cpu()


# ### Zero-shot Image Classification with CLIP
# 

# In[97]:


clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = 'https://hellorfimg.zcool.cn/large/1103065079.jpg'
image = Image.open(requests.get(url, stream=True).raw)


# In[98]:


prompts = [
    "a photo of a lion",
    "a photo of a zebra",
    "a photo of a cat",
    "a photo of an adorable lion cub",
    "a puppy",
    "a lion behind a branch",
]
inputs = processor(
    text=prompts, images=image, return_tensors="pt", padding=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = clip(**inputs)
logits_per_image = outputs.logits_per_image
probabilities = logits_per_image.softmax(dim=1)


# In[99]:


probabilities = probabilities[0].cpu().detach().tolist()


# In[100]:


for prob, prompt in sorted(zip(probabilities, prompts), reverse=True):
    print(f"{100*prob: =2.0f}%: {prompt}")


# ### Zero-shot Image Classification Pipeline
# 

# In[101]:


from transformers import pipeline

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-large-patch14",
    device=device,
)


# In[102]:


scores = classifier(
    image,
    candidate_labels=prompts,
    hypothesis_template="{}",
)


# In[103]:


print(scores)


# ## Solutions
# 
# A big part of learning is putting your knowledge into practice. We strongly suggest not looking at the answers before taking a serious stab at it. Scroll down for the answers.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Exercises
# 
# **1. How does generation work if the `AutoEncoder` model is trained with 16 latent dimensions? Can you compare generations between the model with 16 latent dimensions and the one with just 2?**

# **2. Train the model again with the same parameters we used (just run the code shown in the chapter) but with different random number initialization, and visualize the latent space. Chances are that the shapes and structure are different. Is this something you would expect? Why?**

# **3. How good are the image features extracted by the encoder? Explore it by training a number classifier on top of the encoder.**

# **4. When we trained the VAE, we added the reconstruction and KL-divergence losses. However, both have different scales. What will happen if we give more importance to one versus the other? Can you run a few experiments and explain the results?**

# **5. The VAE we trained only uses two dimensions to represent the mean and the logvar of the distribution. Can you repeat a similar exploration using 16 dimensions?**

# **6. Humans are trained to look at faces and easily identify unrealistic features. Can you try to train an AutoEncoder and a VAE for a dataset containing faces, and see what the results look like? You can start with the [Frey Face dataset](https://cs.nyu.edu/~roweis/data.html) that was used in the VAE paper – it's an homogenous set of monochrome faces from the same person sporting different facial expressions. If you want to be more ambitious, you can try your hand at the [CelebFaces dataset](https://cs.nyu.edu/~roweis/data.html), also hosted on the Hugging Face [Hub](https://huggingface.co/datasets/nielsr/CelebA-faces). Another interesting example could be to try the [Oxford pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), also available on the [Hub](https://huggingface.co/datasets/pcuenq/oxford-pets).**

# ### Challenge
# 

# **7. BLIP-2 for search. The hands-on project on semantic image search is quite challenging, but here's another idea. Can you use the [BLIP-2 model](https://huggingface.co/docs/transformers/main/en/model_doc/blip-2) for similarity tasks, just like we did with CLIP in this chapter? How would you go about it, and how does it compare with CLIP? What other tasks can you solve with BLIP-2?**

# 
