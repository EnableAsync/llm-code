# PPO from Scratch — 大模型 RLHF 训练实现解析

从零实现 PPO（Proximal Policy Optimization，近端策略优化）算法，用于大语言模型（LLM）的 RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）训练。本文档对 `ppo_train.py` 的代码逻辑进行逐模块解析。

![PPO 训练流程](./ppo.png)

## 目录

- [PPO from Scratch — 大模型 RLHF 训练实现解析](#ppo-from-scratch--大模型-rlhf-训练实现解析)
  - [目录](#目录)
  - [背景知识](#背景知识)
  - [整体架构](#整体架构)
  - [核心组件](#核心组件)
    - [PromptDataset：提示词数据集](#promptdataset提示词数据集)
    - [Critic：价值（评论家）模型](#critic价值评论家模型)
    - [Samples 与 Experience：样本与经验](#samples-与-experience样本与经验)
    - [ExperienceBuffer：经验池](#experiencebuffer经验池)
  - [核心算法](#核心算法)
    - [生成样本（Rollout）](#生成样本rollout)
    - [KL 散度近似估计](#kl-散度近似估计)
    - [奖励整形（Reward Shaping）](#奖励整形reward-shaping)
    - [GAE：广义优势估计](#gae广义优势估计)
    - [策略损失（Policy Loss）](#策略损失policy-loss)
    - [价值损失（Value Loss）](#价值损失value-loss)
  - [训练主循环](#训练主循环)
  - [超参数说明](#超参数说明)
  - [运行方式](#运行方式)
    - [环境依赖](#环境依赖)
    - [模型准备](#模型准备)
    - [启动训练](#启动训练)
    - [监控训练](#监控训练)
  - [参考资料](#参考资料)
  - [许可证](#许可证)

## 背景知识

PPO 训练 LLM 时，需要同时维护 **4 个模型**：

| 模型 (Model) | 作用 (Role) | 是否更新参数 |
|---|---|---|
| Actor / Policy Model | 策略模型，即被训练的目标 LLM，负责生成 token | 是 |
| Critic / Value Model | 价值模型，预测每一步动作的未来收益 | 是 |
| Reference Model | 参考模型，与 Actor 初始化相同，用于计算 KL 散度防止偏移 | 否 |
| Reward Model | 奖励模型，对生成结果打分 | 否 |

PPO 的训练流程可以概括为三个阶段的循环：

1. **Rollout（采样）：** Actor 根据 Prompt 生成回复
2. **Make Experience（生成经验）：** 用 4 个模型计算 log_probs、KL、奖励、优势、回报等
3. **Learn（学习）：** 根据 PPO 损失函数更新 Actor 和 Critic

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     训练主循环 train()                    │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │ Prompts      │───>│ Rollout      │───>│ Samples  │ │
│  │ Dataset      │    │ (Actor 推理) │    │          │ │
│  └──────────────┘    └──────────────┘    └─────┬────┘ │
│                                                 │      │
│                                                 ▼      │
│  ┌──────────────────────────────────────────────────┐  │
│  │      generate_experiences(samples)               │  │
│  │  Actor / Ref / Critic / Reward 4 个模型联合推理  │  │
│  │  → log_probs / KL / rewards / advantages         │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           │                            │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │           ExperienceBuffer (经验池)              │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           │                            │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │              train_step(experience)              │  │
│  │   PPO Policy Loss → 更新 Actor                   │  │
│  │   Value Loss      → 更新 Critic                  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### PromptDataset：提示词数据集

```python
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        ...
        for prompt in prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
            self.final_prompts.append(prompt)
```

**作用：** 把原始 Prompt 列表转化为符合模型输入格式的字符串。

**两种模式：**

- `apply_chat_template=True`：使用 Tokenizer 自带的 Chat 模板（适用于 Qwen、Llama-Chat 等指令微调模型）
- `apply_chat_template=False`：在 Prompt 前拼接 `bos_token`

### Critic：价值（评论家）模型

```python
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, num_actions):
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        value_model_output = self.value_head(hidden_state)
        values = value_model_output.squeeze(-1)[:, :-1][:, -num_actions:]
        return values
```

**作用：** 预测每一步（每个生成的 token）的未来期望收益 `V(s_t)`。

**实现要点：**

- 复用 Actor 的 base_model 作为特征提取器
- 外加一个回归头 `value_head`，输出维度为 1
- 输出 `shape: (batch_size, num_actions)`，对应每一个动作（token）的价值

**索引切片解析：**

```python
values = value_model_output.squeeze(-1)[:, :-1][:, -num_actions:]
#                              └─ 去掉最后一位 ─┘ └─ 只保留生成部分 ─┘
```

- `[:, :-1]`：去掉最后一个位置的预测（因果语言模型的标准操作，第 t 个位置预测第 t+1 个 token）
- `[:, -num_actions:]`：只保留 response 部分的价值，prompt 部分不参与训练

### Samples 与 Experience：样本与经验

使用 `@dataclass` 定义两个数据类，分别表示**采样阶段**和**经验生成阶段**的产物：

```python
@dataclass
class Samples:
    seqs: torch.Tensor              # 完整序列：prompt + response
    attention_mask: ...             # 注意力掩码
    action_mask: ...                # 动作掩码（response 部分有效）
    num_actions: ...                # 动作数量（response 长度）
    packed_seq_lens: ...
    response_length: torch.Tensor   # 实际生成长度
    total_length: torch.Tensor      # 总长度

@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor  # Actor 的对数概率
    values: torch.Tensor            # Critic 预测的价值
    returns: ...                    # 回报 R(t)
    advantages: ...                 # 优势 A(t)
    attention_mask: ...
    action_mask: ...
    reward: torch.Tensor            # Reward Model 打分
    response_length: ...
    total_length: ...
    num_actions: ...
    kl: ...                         # 与 Reference Model 的 KL 散度
```

### ExperienceBuffer：经验池

```python
class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences):
        # 把 Experience 对象拆解成字典，逐条加入 buffer
        ...
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
```

**作用：** 类似 DQN 中的 Replay Buffer，存储采样得到的经验，供训练阶段反复使用（PPO 的 `max_epochs` 决定一批经验被复用多少轮）。

**容量管理：** 超出 `limit` 后，丢弃最早的经验（FIFO）。

## 核心算法

### 生成样本（Rollout）

```python
def generate_samples(prompts, model, max_length, max_new_tokens,
                     n_samples_per_prompt, micro_rollout_batch_size):
    samples_list = []
    model.eval()
    # 每个 prompt 生成 n_samples_per_prompt 个样本（增加多样性）
    all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in prompts], [])

    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i:i+micro_rollout_batch_size]
        inputs = actor_tokenizer(prompts, padding='max_length', max_length=max_length,
                                 truncation=True, return_tensors='pt')
        seqs = model.generate(**inputs.to(device),
                              max_new_tokens=max_new_tokens,
                              eos_token_id=eos_token_id,
                              pad_token_id=pad_token_id)
        # 处理长度对齐：不足则补 pad，超过则截断
        ...
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        ans = seqs[:, input_ids.size(1):]
        action_mask = ans.ne(pad_token_id).to(dtype=torch.long)
        ...
```

**关键点：**

1. **多样性采样：** 一个 prompt 复制 `n_samples_per_prompt` 份，让模型生成多个不同回复
2. **micro batch：** 限制单次推理的 batch size，避免显存溢出（生成阶段需要 4 个模型同时占用显存）
3. **左填充：** `actor_tokenizer.padding_side = 'left'`，保证 `generate` 时所有样本的生成位置对齐
4. **action_mask：** 标记 response 中哪些 token 是真实生成的（非 pad）

### KL 散度近似估计

```python
def compute_approx_kl(log_probs, ref_log_probs, action_mask=None):
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio
```

**作用：** 计算 Actor 与 Reference Model 在每个 token 上的 KL 散度，用于约束 Actor 不要偏离原始模型太远（防止 reward hacking）。

**近似公式：** `KL ≈ log_probs - ref_log_probs`（这是单样本估计，非真实 KL）

### 奖励整形（Reward Shaping）

```python
def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    kl_divergence_estimate = -kl_ctl * kl
    rewards = kl_divergence_estimate

    ends = action_mask.sum(1)
    reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)

    batch_size = r.size(0)
    for j in range(batch_size):
        rewards[j, :ends[j]][-1] += reward_clip[j, 0]

    return rewards
```

**核心思路：** 把 Reward Model 给出的**整段序列的单一分数**转化为**逐 token 奖励**。

**奖励组成：**

- **过程奖励（每个 token）：** `-kl_ctl * KL`，惩罚偏离 Reference Model 的行为
- **结果奖励（最后一个 token）：** Reward Model 打分（裁剪到 `[-clip_reward_value, clip_reward_value]`）

**结构示意：**

```
位置:      1     2     3   ...  T-1    T (最后一个有效 token)
奖励: -β*KL -β*KL -β*KL ... -β*KL  (-β*KL + R)
                                         └─ 只在最后位置加上 RM 分数 ─┘
```

### GAE：广义优势估计

```python
def get_advantages_and_returns(values, rewards, action_mask, gamma, lambd):
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)

    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns
```

**核心公式：**

- 单步 TD 误差：`δ(t) = R(t) + γ·V(t+1) − V(t)`
- 朴素优势：`A(t) = δ(t)`
- **GAE：** `A(t) = δ(t) + γλ·A(t+1)`
- 回报：`Returns(t) = A(t) + V(t)`

**反向递推的原因：**

由于 `A(t)` 依赖于 `A(t+1)`，所以从最后一个时刻 `T` 开始倒序计算：

- 终止条件：`A(T+1) = 0, V(T+1) = 0`
- 因此 `A(T) = R(T) − V(T)`
- 然后 `A(T-1) = δ(T-1) + γλ·A(T)`，依次类推

**γ 与 λ 的含义：**

| 参数 | 含义 |
|---|---|
| `gamma (γ)` | 折扣因子，控制对未来奖励的重视程度 |
| `lambd (λ)` | GAE 平滑系数，在偏差和方差之间权衡（λ=0 退化为 TD，λ=1 退化为 Monte Carlo） |

### 策略损失（Policy Loss）

```python
def compute_policy_loss(log_probs, old_log_probs, advantages,
                        action_mask=None, clip_eps=0.2):
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2)
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()
```

**PPO 的核心 —— Clipped Surrogate Objective：**

```
L^CLIP(θ) = E[ min( r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A ) ]
```

其中：

- `r(θ) = π_θ(a|s) / π_θ_old(a|s)` 是新旧策略的概率比
- `clip(...)` 把 ratio 限制在 `[1-ε, 1+ε]` 范围内，**防止策略更新步长过大导致训练崩溃**
- 取 `min` 是为了构造一个保守的下界

### 价值损失（Value Loss）

```python
def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps=None):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2
    ...
```

**作用：** 拟合 Critic 的价值预测，使其逼近真实回报 `returns`。

**支持两种模式：**

- 不裁剪：标准 MSE 损失 `(V − R)²`
- 裁剪：限制 `V` 偏离 `V_old` 不超过 `clip_eps`，与 Policy Loss 形式上对称

## 训练主循环

```python
def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):                      # 外层：训练轮数
        for rand_prompts in prompts_dataloader:          # 中层：从数据集取 prompts
            # 1. Rollout：Actor 生成回复
            samples = generate_samples(rand_prompts, actor_model, ...)
            # 2. Make Experience：4 个模型联合推理，得到 log_probs/KL/rewards/advantages
            experiences = generate_experiences(samples)
            buffer.append(experiences)

            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size,
                                    shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()

            # 3. Learn：同一批经验复用 max_epochs 轮，更新 Actor 和 Critic
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1

            buffer.clear()
            torch.cuda.empty_cache()
```

**train_step 详解：**

```python
def train_step(experience, steps):
    # === 更新 Actor ===
    actor_model.train()
    optimizer_actor.zero_grad()

    logits = actor_model(sequences, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs,
                                      advantages, action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()

    # === 更新 Critic ===
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
```

**注意 log_probs 的提取技巧：**

```python
log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)         # (B, L-1, V)
log_probs_labels = log_probs.gather(
    dim=-1, index=sequences[:, 1:].unsqueeze(-1))            # (B, L-1, 1)
action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]  # (B, num_actions)
```

- `logits[:, :-1, :]`：第 t 个位置预测第 t+1 个 token
- `sequences[:, 1:]`：右移一位作为标签
- `gather`：从 vocab 维度抽取对应 label 的概率
- `[:, -num_actions:]`：只保留 response 部分

## 超参数说明

| 参数 (Parameter) | 默认值 (Value) | 说明 (Description) |
|---|---|---|
| `episodes` | 3 | 总训练轮数 |
| `max_epochs` | 5 | 一批经验复用的训练轮数 |
| `rollout_batch_size` | 8 | 单次从数据集中取多少 prompt 用于 rollout |
| `micro_rollout_batch_size` | 2 | 单次推理的 batch size（受显存限制） |
| `n_samples_per_prompt` | 2 | 每个 prompt 生成多少个样本 |
| `max_new_tokens` | 50 | 单次生成的最大 token 数 |
| `max_length` | 256 | Prompt 最大长度 |
| `micro_train_batch_size` | 2 | 训练阶段的 batch size |
| `kl_ctl` | 0.1 | KL 惩罚系数 β |
| `clip_reward_value` | 0.2 | Reward 的裁剪阈值 |
| `gamma` | 0.1 | GAE 折扣因子 γ |
| `lambd` | 0.2 | GAE 平滑系数 λ |
| `clip_eps`（policy） | 0.2 | PPO 概率比裁剪阈值 ε |
| Actor / Critic 学习率 | 5e-5 | Adam 优化器 |

> **提示：** `gamma=0.1` 和 `lambd=0.2` 偏小，这里仅作演示。实际 RLHF 训练中常见取值为 `gamma=1.0, lambd=0.95`，可根据实验调整。

## 运行方式

### 环境依赖

```bash
pip install torch transformers tensorboard
```

### 模型准备

代码中默认加载以下模型（路径需根据实际环境修改）：

```python
actor_model  = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
ref_model    = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
reward_model = AutoModelForSequenceClassification.from_pretrained(
                   '/home/user/Downloads/reward-model-deberta-v3-large-v2')
critic_model = Critic(actor_model.base_model)
```

- **Actor / Reference Model：** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Reward Model：** [reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)
- **Critic Model：** 复用 Actor 的 base_model，外加回归头

### 启动训练

```bash
python ppo_train.py
```

### 监控训练

训练过程会写入 TensorBoard 日志到 `./runs` 目录：

```bash
tensorboard --logdir ./runs
```

可观察 `policy_loss` 和 `value_loss` 的变化趋势。

## 参考资料

- [Schulman et al., 2017 — Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Schulman et al., 2015 — High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438)
- [Ouyang et al., 2022 — Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)：本代码部分实现思路参考自该项目

## 许可证

本代码仅用于教学与研究目的。
