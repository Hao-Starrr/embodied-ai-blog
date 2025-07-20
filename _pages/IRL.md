---
layout: article
title: IRL
permalink: /irl/
---


逆强化学习（**Inverse Reinforcement Learning**，简称 **IRL**）是一种在**强化学习（RL）**框架中的研究方法，它的核心目标是：



​	**已知一个智能体的行为轨迹，推测出它背后隐含的奖励函数（或目标）。**



------



**🚗 打个简单的比方：**



假设你是个驾校老师，看到了一个老司机在不同情况下开车（比如加速、刹车、避让等）。你不知道他脑子里具体的**“开车准则”**（也就是奖励函数），但你能看到他的行为。那么你就可以通过 逆强化学习，尝试**反推**他做这些操作是为了优化什么目标（例如安全第一、时间最短、油耗最低等）。



------



**✅ 正向 vs 逆向**

| **类型**              | **目标**                 | **已知**                 | **结果**         |
| --------------------- | ------------------------ | ------------------------ | ---------------- |
| **强化学习（RL）**    | 学习如何行动以最大化奖励 | 奖励函数已知             | 策略（如何行动） |
| **逆强化学习（IRL）** | 学习奖励函数本身         | 行为轨迹已知（专家演示） | 奖励函数         |





------



**📦 逆强化学习的主要用途**

​	•	🧑‍🏫 **模仿学习（Imitation Learning）**：希望机器人模仿人类专家的操作行为，但不只模仿表面动作，而是学习**背后的目的**。

​	•	⚙️ **人机协作**：理解人类在协作任务中追求的目标，以便更好协作。

​	•	🤖 **机器人规划**：让机器人理解人类在复杂任务中的偏好（比如照顾老人时的温柔 vs 高效）。

​	•	🧠 **认知科学**：用于推断动物或人类的行为动机。



------



**🧮 数学上怎么做？**



给定专家的轨迹（比如一系列状态-动作对），IRL尝试找到一个奖励函数，使得这些轨迹是“最优”的。这通常涉及：

​	1.	建立马尔可夫决策过程（MDP）的结构：<S, A, T, γ>，但**没有奖励函数 R**。

​	2.	假设专家执行的是最优策略。

​	3.	利用观测到的行为，优化出一个使专家轨迹最优的 R。



------



**📘 常见方法有哪些？**

​	•	**MaxEnt IRL**（Maximum Entropy IRL） – 经典方法，引入最大熵原则，鼓励保留行为的不确定性。

​	•	**GAIL**（Generative Adversarial Imitation Learning）– 使用GAN框架对抗学习，模仿行为分布。

​	•	**AIRL**（Adversarial IRL）– 从GAIL演变而来，可以恢复出真实的奖励函数。



------



我们现在来看示例：



------



**🌍 场景设定：真实机器人轨迹**



我们假设机器人在一个 2D 空间中执行任务，比如抓取、推物、避障等。机器人轨迹以 state = [x, y] 或 [x, y, θ] 表示。我们采集了多段专家演示的状态序列（没有动作 label），希望通过 **神经网络**来学习出隐含的奖励函数。



------



**🧠 整体流程（Neural IRL）**



我们使用 **Maximum Entropy IRL + 神经网络** 表达奖励函数 R(s)，训练流程如下：

​	1.	使用神经网络拟合 R_\theta(s)

​	2.	用 soft value iteration 计算策略 π(a|s)

​	3.	用 π(a|s) 推出期望状态访问频率

​	4.	最小化专家分布和策略分布的差异



------



**🧪 实践：用 PyTorch 实现 Neural IRL（简化版本）**



**🎯 数据设定**

​	•	状态是 2D 的：[x, y]

​	•	动作是离散的：上下左右

​	•	环境是一个连续空间中的机器人导航任务

​	•	专家演示是多个状态轨迹，比如：

```
expert_trajs = [
    [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.2], [0.8, 0.4]],
    [[0.1, 0.1], [0.3, 0.1], [0.5, 0.2], [0.7, 0.3], [0.9, 0.5]],
    ...
]
```





------



**✅ 神经网络 IRL 实现（核心代码）**

```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 模拟数据（专家轨迹）
expert_trajs = []
for _ in range(10):
    traj = []
    for t in range(10):
        x = t * 0.1
        y = 0.5 * x + 0.05 * np.random.randn()
        traj.append([x, y])
    expert_trajs.append(traj)

# 状态空间采样
def sample_states(n=1000):
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

# 奖励函数网络
class RewardNet(nn.Module):
    def __init__(self, input_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)

# 经验状态分布（专家的访问频率）
def empirical_state_distribution(trajs):
    states = []
    for traj in trajs:
        states += traj
    states = torch.tensor(states, dtype=torch.float32)
    return states

# 训练 IRL：用最大熵 IRL 的思想
reward_net = RewardNet()
optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)

# 训练
for epoch in range(500):
    # 采样状态
    states = sample_states(1024)
    rewards = reward_net(states)

    # 从专家轨迹中采样状态
    expert_states = empirical_state_distribution(expert_trajs)
    expert_rewards = reward_net(expert_states)

    # Loss: MaxEnt IRL 的差值
    loss = -(expert_rewards.mean() - torch.logsumexp(rewards, dim=0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"[{epoch}] loss: {loss.item():.4f}")

# ✅ 可视化 reward 函数
with torch.no_grad():
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(0, 1, 100),
        torch.linspace(0, 1, 100),
        indexing='xy'
    )
    grid_states = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    grid_rewards = reward_net(grid_states).reshape(100, 100)

plt.imshow(grid_rewards.numpy(), extent=(0, 1, 0, 1), origin='lower', cmap='plasma')
plt.colorbar()
plt.title("Learned Reward Function")
plt.show()
```





------



**📌 结果解释**

​	•	神经网络学出的 reward heatmap 应该在专家路径附近有高值，其他地方低值。

​	•	换句话说，它学会了“哪些状态是好状态”，而没有直接模仿动作。

​	•	这个 reward 可以用于后续强化学习，再训练一个 policy 去最大化它。



------



**📚 可扩展方向**

​	1.	✅ 把动作也作为输入 R(s, a) 来学习奖励

​	2.	✅ 用 GAIL（对抗 IRL）代替最大熵 IRL

​	3.	✅ 用真实机器人轨迹（来自 ROS、Franka 或 UR 数据集）

​	4.	✅ 加入动作模型和真实动力学推理



------



你想我们下一步深入哪个方向？比如用 GAIL，还是你有现成轨迹数据我们可以套这个流程跑？