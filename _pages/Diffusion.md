---
layout: article
title: Diffusion
permalink: /diffusion/
---


# 先看视频



【简单易懂Diffusion模型综述 - 基础算法详解】 https://www.bilibili.com/video/BV1TP4y1Q7qJ/?share_source=copy_web&vd_source=0261590b30f63d16142104bb154164d9

【Why Does Diffusion Work Better than Auto-Regression?(为什么扩散比自回归更有效？)】 https://www.bilibili.com/video/BV1zb42187Ec/?share_source=copy_web&vd_source=0261590b30f63d16142104bb154164d9

【大白话01一文理清 Diffusion Model 扩散模型 | 原理图解+公式推导】 https://www.bilibili.com/video/BV1xih7ecEMb/?share_source=copy_web&vd_source=0261590b30f63d16142104bb154164d9



# diffusion是什么？



## **✅ Diffusion 的目标是学习图像分布**

我们有很多真实图像样本 $x_0 \sim p_{\text{data}}(x)$，x表示图像，这些图像本身可能来自很复杂的分布，比如：

- 有猫、有狗、有房子
- 有颜色、有纹理、有边缘
- 这些内容在图像像素空间中有 **统计相关性**

## **🧠 所以我们想学这个分布** $p_{\text{data}}(x)$

学到了这个分布就能自己生成看上去还不错的图片了。

但是这个分布是我们不知道的，直接建模它太难了，于是 diffusion 做了一件特别巧妙的事：



------



### **🧊 1.** **构造一个容易的分布：高斯噪声**

我们知道一个很简单的分布：标准正态分布 $\mathcal{N}(0, I)$。

它没有任何结构——纯噪声。

### **🔁 2.** **一步步把真实图像“毁掉”，变成高斯**

这个过程是我们自己设计出来的，叫做 **前向扩散过程**（Forward Diffusion）：

$x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T \approx \mathcal{N}(0, I)$

每一步我们人为加噪。最终你会把结构丰富的图像变成结构全无的噪声。

这一步是有 closed-form 的（可以直接计算），不需要训练。

### **🧑‍🎓 3.** **训练一个神经网络做“去噪”**

这时候我们要训练一个神经网络来“逆转”这个过程，让它学会：

$x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0$

但这个过程本身不是我们有现成数据的，而是我们通过训练让神经网络**学会如何一步步还原结构**。

为了训练这个网络，大家设计了一个很巧的目标函数，就是让网络预测你加进去的噪声 $\epsilon$，从而间接学会还原图像。

### **🤖 所以你训练的神经网络其实在做：**

- 给它一个 noisy 图像 $x_t$
- 给它一个时间步 $t$
- 它要预测的是这个图像中有多少噪声（也就是你加了多少扰动）
- 这样你就能一步步“反着走”回去，还原出原始图像

## **✅ 最终目标：生成图像！**

一旦你训练好了这个“去噪神经网络”，你就可以：

1. 从 $x_T \sim \mathcal{N}(0, I)$ 采一个随机噪声
2. 反复应用网络，一步步去噪
3. 最终你得到一个新的 $x_0$，它是来自于学到的图像分布 $p_{\text{data}}(x)$。这就完成了图像生成。

## **📌 所以 diffusion 的核心结构是：**

- 设计 Forward 过程（人为加噪，模拟随机性）
- 训练神经网络学会 Reverse（一步步去噪）
- 推理的时候从高斯噪声出发，用神经网络“生成图像”



------



# 怎么做呢？



## **🧠 回顾：我们的目标是学会“反向还原图像”**

我们设计了一个前向过程，把真实图像一步步加噪：

$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$

最终到第 T 步变成纯高斯噪声 $x_T \sim \mathcal{N}(0, I)$。

我们要训练的神经网络，是学会“从 noisy 图像中逐步去噪”，也就是估计：

$p_\theta(x_{t-1} \mid x_t)$



### **基本变量定义：**

| **符号**                                | **含义**                                |
| --------------------------------------- | --------------------------------------- |
| $x_0$                                   | 原始图像（来自训练集）                  |
| $x_t$                                   | 第 t 步加噪后的图像（包含部分噪声）     |
| $T$                                     | 总共加噪的步数，比如 1000 步            |
| $\beta_t$                               | 第 t 步加进去的噪声强度（噪声调节因子） |
| $\alpha_t = 1 - \beta_t$                | 每一步保留原图的比例                    |
| $\bar{\alpha}t = \prod{s=1}^t \alpha_s$ | 到第 t 步为止累计保留原图的比例         |



## **✅ 训练方式一：** **预测噪声** $\epsilon$**（最主流做法）**

最常见的 diffusion（比如 DDPM）是用 **噪声预测** 的方式来训练网络。

### **🧪 数据构造（采样训练样本）**

对于每一张图像 $x_0 \sim \text{真实图像}$，我们做以下步骤：

1. 随机采样一个时间步 $t \sim \{1, \dots, T\}$
2. 采一个随机噪声 $\epsilon \sim \mathcal{N}(0, I)$
3. 按照 closed-form 的加噪公式得到 $x_t$：

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

其中 $\bar{\alpha}t = \prod{s=1}^{t}(1 - \beta_s)$ 是累计的保持比例。

- $x_0$：原始图像；
- $\epsilon \sim \mathcal{N}(0, I)$：标准高斯噪声；
- $\sqrt{\bar{\alpha}_t}$：当前保留图像信息的比例；
- $\sqrt{1 - \bar{\alpha}_t}$：当前混入噪声的比例。

这表示：**在任意时间步** t**，我们都能直接采样对应的 noisy 图像** $x_t$。

### **🎯 神经网络目标：预测噪声**

我们训练一个神经网络 $\epsilon_\theta(x_t, t)$，目标是预测我们加进去的噪声 $\epsilon$。训练目标是：

$L(\theta) = \mathbb{E}{x_0, \epsilon, t} \left[ \left\| \epsilon\theta(x_t, t) - \epsilon \right\|^2 \right]$

这样网络学会了：给定一个 noisy 图像和时间步，它能告诉你这个图像里藏了多少噪声。

- $\theta$：网络的可学习参数；

- $x_0$：从数据集中采样的一张真实图像；
- $\epsilon \sim \mathcal{N}(0, I)$：我们人为加进去的随机噪声；
- $t \sim \text{Uniform}(1, T)$：随机选取一个时间步；
- $x_t$：由前向过程公式计算出来的 noisy 图像；
- $\epsilon_\theta(x_t, t)$：模型的预测输出，一个与 $x_t$ 同形状的张量；
- 目标：**让模型预测出的噪声尽量接近我们真实加进去的噪声** $\epsilon$。



## **🤖 网络结构怎么设计？**

### **1.** **输入是两个东西**

- $x_t$：一张 noisy 图像，形状和原始图像一样（比如 $3 \times 64 \times 64$）
- t：一个整数时间步，表示当前图像中混入了多少噪声

### **2.** **网络结构一般是 U-Net**（特别适合图像处理）

- 输入：noisy 图像 $x_t$
- 时间步 t 会被编码成一个时间向量（embedding），加入到每一层
- 使用 skip-connection，保留高分辨率信息
- 输出和输入同样大小，表示每个像素的噪声值（和 $\epsilon$ 同型）

这就是 diffusion 模型的核心网络。

### **3.** **时间步** t **的处理方式**

时间是一个整数 $t \in \{1, \dots, T\}$，不能直接送给网络。一般的做法是：

- 用一个 **位置编码（Positional Encoding）** 把 t 编成一个向量；
- 加入到网络的每一层（比如用 FiLM 或直接加到激活上）；

这样网络就知道当前是哪一步，还原多少结构。

### **输出：**

- 一个和 $x_t$ 同样形状的张量；
- 表示每个像素的噪声预测值；



## **✅ 推理阶段怎么用这个网络？**

训练完网络后，我们可以用它一步步去噪：

1. 从 $x_T \sim \mathcal{N}(0, I)$采样
2. 用神经网络预测噪声 $\hat{\epsilon}_\theta(x_T, T)$
3. 用下面的公式还原 $x_{T-1}$：

$x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \beta_t \cdot \hat{\epsilon}_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t} \right) + \text{some noise}$

重复上面的步骤直到 x_0，得到一张新图像。



## **🔁 总结一下：**

| **步骤** | **内容**                                       |
| -------- | ---------------------------------------------- |
| 训练目标 | 让神经网络预测图像里的噪声                     |
| 网络结构 | U-Net + 时间步编码                             |
| 输入输出 | 输入 noisy 图像 $x_t$ 和时间步 t，输出预测噪声 |
| 损失函数 | MSE：预测的噪声 vs 实际加进去的噪声            |
| 推理过程 | 从噪声出发，逐步去噪恢复图像                   |





# 代码练习



我们就通过一个**从头训练 diffusion 模型的完整 PyTorch demo**，一步一步走一遍流程。

为了简单清晰，我们使用 **MNIST（28x28 灰度图像）** 作为训练数据，使用经典的 **DDPM（denoising diffusion probabilistic model）** 框架。

------



## **🧩 整体结构分为 4 步：**

1. 加噪函数（前向过程）
2. 网络结构（一个小 U-Net）
3. 损失函数（噪声预测）
4. 训练循环



## **🧪 Step 0: 安装依赖（只需 PyTorch 和 torchvision）**

```
pip install torch torchvision
```

## **🧠 Step 1: 前向加噪函数**

```python
import torch
import torch.nn.functional as F
import numpy as np

# beta schedule（噪声强度）线性从小到大
T = 300  # 时间步数
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 加噪函数 q(x_t | x_0)
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)
    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
```

## **🧱 Step 2: 一个简单的 U-Net 网络结构（适配 MNIST）**

```python
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1 + 1, 32, 3, padding=1),  # 图像 + 时间通道
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # 输出与输入同形状
        )

    def forward(self, x, t):
        # 把时间步 t 编码成一张和图像一样大小的通道图
        t = t.float() / T
        t = t.view(-1, 1, 1, 1).expand(x.shape[0], 1, 28, 28)
        x = torch.cat([x, t], dim=1)  # 拼接时间通道
        return self.net(x)
```

## **🎯 Step 3: 训练 loss 函数（预测噪声）**

```python
def diffusion_loss(model, x_0, t):
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, noise)
    pred_noise = model(x_t, t)
    return F.mse_loss(pred_noise, noise)
```

## **🔁 Step 4: 训练循环**

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载 MNIST 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x * 2. - 1.  # 归一化到 [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 初始化模型
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练主循环
for epoch in range(10):
    for x, _ in dataloader:
        x = x.to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device).long()
        loss = diffusion_loss(model, x, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```



------



## **🎨 (Bonus) 推理采样函数（生成图像）**

```python
@torch.no_grad()
def sample(model, n_samples=16):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)

    for t_ in reversed(range(T)):
        t = torch.full((n_samples,), t_, device=device, dtype=torch.long)
        noise_pred = model(x, t)

        beta_t = beta[t].view(-1, 1, 1, 1).to(device)
        alpha_t = alpha[t].view(-1, 1, 1, 1).to(device)
        alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1).to(device)

        if t_ > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = 1 / torch.sqrt(alpha_t) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) + torch.sqrt(beta_t) * noise

    return x
```



------



## **✅ 运行 sample 后你可以用 matplotlib 展示生成图像：**

```python
import matplotlib.pyplot as plt

samples = sample(model, 16).cpu()
grid = samples.view(4, 4, 28, 28).permute(0, 2, 1, 3).reshape(4*28, 4*28)
plt.imshow(grid, cmap='gray')
plt.axis('off')
plt.show()
```



## **📌 总结**

你现在有了一个完整的 diffusion 模型训练管线：

1. 从真实图像构造 noisy 图像；
2. 用神经网络预测噪声；
3. 用 MSE loss 训练网络；
4. 用网络逐步还原图像。





