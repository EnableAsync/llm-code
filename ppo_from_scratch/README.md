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
      - [为什么 Rollout 必须使用左填充？](#为什么-rollout-必须使用左填充)
    - [KL 散度近似估计](#kl-散度近似估计)
    - [奖励整形（Reward Shaping）](#奖励整形reward-shaping)
    - [GAE：广义优势估计](#gae广义优势估计)
    - [策略损失（Policy Loss）](#策略损失policy-loss)
    - [价值损失（Value Loss）](#价值损失value-loss)
  - [训练主循环](#训练主循环)
    - [Actor 与 Critic 是交替训练的吗？](#actor-与-critic-是交替训练的吗)
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

#### 为什么 Rollout 必须使用左填充？

代码中通过这一行设置了左填充：

```python
actor_tokenizer.padding_side = 'left'
```

核心原因：**`model.generate()` 是从输入序列的最右端开始续写的**。

**右填充的问题（错误示范）：**

如果使用右填充，batch 里不同长度的 prompt 会变成这样：

```
样本 1（短 prompt）：[P1] [P2] [P3] [PAD] [PAD] [PAD] | ← 从这里开始生成
样本 2（长 prompt）：[Q1] [Q2] [Q3] [Q4]  [Q5]  [Q6]  | ← 从这里开始生成
```

模型会接在 `[PAD]` 后面继续生成，相当于在 pad token 之后续写——**生成的内容前面会先续上一堆 `<pad>`**，模型也无法基于真实 prompt 的最后一个 token 进入生成状态。

**左填充（正确做法）：**

```
样本 1（短 prompt）：[PAD] [PAD] [PAD] [P1] [P2] [P3] | ← 对齐到右端开始生成
样本 2（长 prompt）：[Q1]  [Q2]  [Q3]  [Q4] [Q5] [Q6] | ← 对齐到右端开始生成
```

所有样本的**真实 prompt 末尾都对齐到了序列的最右端**，于是：

1. `generate()` 从同一个相对位置开始往后续写
2. 每个样本生成的 token 在 batch 中处于**同一列**，便于后续切片：
   ```python
   ans = seqs[:, input_ids.size(1):]  # 直接截取生成部分
   ```
3. `attention_mask` 会让模型自动忽略左边的 pad token，不影响注意力计算

**对 action_mask 的影响：**

代码中 `action_mask = ans.ne(pad_token_id)` 这一步之所以能正确工作，也依赖左填充——因为生成段（response）部分如果出现 pad，那一定是 EOS 之后模型停止生成时填充的，而不是 prompt 区域的 pad 串扰过来。

**一句话总结：**

> 左填充是为了让 batch 中所有 prompt 的「末尾真实 token」对齐到同一位置，使 `generate()` 能从统一的起点开始续写，并保证生成结果在 batch 维度上整齐对应。

> **补充：** 训练阶段输入和标签是已经存在的完整序列（prompt + response），不需要"接着最后一个 token 续写"，因此左填充在训练阶段不是必需的。但本代码复用了同一个 tokenizer，整个流程都保持左填充也不会出错。

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

#### 真实的优势是怎么得到的？

**核心结论：代码里得到的并不是"真正的"优势，而是 GAE 估计出来的近似优势。**

**优势的理论定义：**

在强化学习中，"真实"的优势函数定义为：

```
A(s_t, a_t) = Q(s_t, a_t) − V(s_t)
            = E[ R_t + γR_{t+1} + γ²R_{t+2} + ... ] − V(s_t)
```

也就是「在状态 s 下采取动作 a 比平均水平好多少」。

**问题：** 我们既拿不到精确的 Q，也拿不到精确的期望，**只能估计**。这就是 GAE 要解决的问题。

**本代码中优势的产生链路：**

```
┌──────────────┐
│ Actor 模型   │── log_probs ──┐
└──────────────┘                │
                                ▼
┌──────────────┐         ┌──────────────┐
│ Reference 模型│── ref ─>│ compute_     │── KL（逐 token）
└──────────────┘         │ approx_kl    │
                         └──────────────┘
                                │
                                ▼
┌──────────────┐         ┌──────────────────┐
│ Reward 模型  │── r ───>│ compute_rewards  │── rewards（逐 token）
└──────────────┘         │  = -β·KL +       │       │
                         │   末位加 RM 分数 │       │
                         └──────────────────┘       │
                                                    │
┌──────────────┐                                    ▼
│ Critic 模型  │── values（逐 token）─> ┌──────────────────────────┐
└──────────────┘                        │get_advantages_and_returns│
                                        │     (GAE 反向递推)        │
                                        └──────────────────────────┘
                                                    │
                                                    ▼
                                            advantages, returns
```

对应代码位于 `generate_experiences`：

```python
kl = compute_approx_kl(action_log_probs, ref_action_log_probs, action_mask)      # ① KL
rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2) # ② 逐 token 奖励
advantages, returns = get_advantages_and_returns(                                # ③ GAE
    value, rewards, action_mask, gamma=0.1, lambd=0.2)
```

**关键三步详解：**

**第一步：把"序列级奖励"转成"逐 token 奖励"**

Reward Model 给整段回复打**一个分数**，但 PPO 需要每个 token 都有奖励。`compute_rewards` 做了奖励整形：

```
位置:      0      1      2     ...    T-1     T (最后有效 token)
奖励:   -β·KL  -β·KL  -β·KL   ...   -β·KL  (-β·KL + R_clip)
        └────── 过程奖励：抑制偏离 Reference ──────┘└── 结果奖励 ──┘
```

- **过程奖励 (process reward)：** 每个 token 都给 `-β·KL`，作用是惩罚 Actor 偏离 Reference Model
- **结果奖励 (outcome reward)：** 只在最后一个有效 token 上加上 RM 分数

> 这其实是「结果奖励模型 + KL 正则」的稀疏奖励范式，OpenAI 的 InstructGPT 采用的就是这个套路。

**第二步：Critic 给出每一步的基线 V(s_t)**

```python
value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
# shape: (batch_size, num_actions)
```

`values[t]` 表示「站在第 t 个 token 这个状态，未来期望能拿到多少总奖励」。它是**学出来的**，不是真值。

**第三步：GAE 反向递推得到优势**

数学上做的是：

```
δ(t) = R(t) + γ·V(t+1) − V(t)            ← 单步 TD 误差
A(t) = δ(t) + γλ·A(t+1)                   ← GAE 递推
A(T) = R(T) − V(T)                        ← 终止条件（最后一步）
```

只有从后往前算，才能让每一步用上"未来"的优势。

**三种优势估计的关系：**

GAE 的精妙之处在于**用一个超参数 λ 在两种极端估计之间插值**：

| λ 取值 | 等价于 | 偏差 (Bias) | 方差 (Variance) |
|---|---|---|---|
| **λ = 0** | 单步 TD：`A(t) = R(t) + γV(t+1) − V(t)` | 高（依赖 Critic 准不准） | 低 |
| **λ = 1** | Monte Carlo：`A(t) = Σ γ^k·R(t+k) − V(t)` | 低 | 高（轨迹长就抖） |
| **0 < λ < 1** | GAE：在两者之间平滑插值 | 适中 | 适中 |

> 所以 GAE 给的不是「真值」，是**偏差和方差权衡之后的一个估计量**。RLHF 里通常用 `gamma=1.0, lambd=0.95`。

**优势 vs. 回报：关系一目了然**

代码最后一行：

```python
returns = advantages + values
```

这一行揭示了二者关系：

```
A(t)        = (估计的真实回报) − V(t)        ← 优势
Returns(t)  = A(t) + V(t) = (估计的真实回报)  ← 回报
```

- **`returns` 用来训 Critic**（让 V(t) 去拟合 returns）
- **`advantages` 用来训 Actor**（告诉策略"这个动作比基线好多少"）

两个量来自同一份 GAE 计算，但服务于不同的损失。

**一句话总结：**

> 代码中的优势不是真实优势，而是用「KL 正则 + RM 末位奖励」拼出逐 token 奖励，再让 Critic 提供基线，最后通过 GAE 反向递推得到的估计值——它在偏差与方差之间做权衡，是 PPO 训练 LLM 的标准做法。

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

**train_step 详解（逐行注释）：**

```python
def train_step(experience, steps):
    # ============================================================
    # 第一阶段：更新 Actor（策略模型）
    # ============================================================
    actor_model.train()              # 切换到训练模式（启用 dropout 等）
    optimizer_actor.zero_grad()      # 清空上一步累积的梯度

    # ── 从 experience 中取出训练所需字段 ──
    sequences            = experience.seqs               # 完整序列 (B, L) = prompt + response
    old_action_log_probs = experience.action_log_probs   # 采样时 Actor 给出的旧 log π_old(a|s)
    advantages           = experience.advantages         # GAE 估计的优势 A(t)
    num_actions          = experience.num_actions        # response 长度（动作数量）
    attention_mask       = experience.attention_mask     # 注意力掩码（pad=0）
    action_mask          = experience.action_mask        # 动作掩码（仅 response 部分=1）
    old_values           = experience.values             # 采样时 Critic 给出的旧 V_old(t)
    returns              = experience.returns            # GAE 算出的回报，用于 Critic 拟合

    # ── 前向传播：让"当前 Actor"重新计算 log π_new(a|s) ──
    logits = actor_model(
        sequences,
        attention_mask=attention_mask
    ).logits                          # (B, L, V) —— 每个位置的词表 logits

    # 因果 LM 的标准操作：第 t 个位置的 logits 用来预测第 t+1 个 token
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)        # (B, L-1, V)

    # gather：从 vocab 维度抽取实际 token 对应的 log 概率
    # sequences[:, 1:] 是右移一位的"标签"
    log_probs_labels = log_probs.gather(
        dim=-1,
        index=sequences[:, 1:].unsqueeze(-1)                    # (B, L-1, 1)
    )

    # 切片：只保留 response 部分的 log 概率（prompt 部分不参与策略更新）
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]  # (B, num_actions)

    # ── 计算 PPO 裁剪策略损失 ──
    # ratio = exp(log π_new - log π_old) = π_new / π_old
    # L_CLIP = -min( ratio·A,  clip(ratio, 1-ε, 1+ε)·A )
    policy_loss = compute_policy_loss(
        action_log_probs,             # 当前策略给出的新 log 概率
        old_action_log_probs,         # 采样时的旧 log 概率（已 detach，不参与求导）
        advantages,                   # GAE 优势
        action_mask=action_mask       # 屏蔽 pad 位置
    )

    policy_loss.backward()           # 反向传播：梯度只流向 Actor
    optimizer_actor.step()           # 更新 Actor 参数

    writer.add_scalar("policy_loss", policy_loss.item(), steps)  # 记录到 TensorBoard

    # ============================================================
    # 第二阶段：更新 Critic（价值模型）
    # ============================================================
    critic_model.train()             # Critic 切换到训练模式
    optimizer_critic.zero_grad()     # 清空 Critic 的梯度

    # ── 用"当前 Critic"重新计算每个 token 的价值 V_new(t) ──
    values = critic_model.forward(
        sequences,
        attention_mask,
        num_actions                  # 内部会切片只保留 response 部分
    )                                 # (B, num_actions)

    # ── 计算价值损失（MSE 或裁剪 MSE）──
    # L_V = (V_new - returns)²
    # 让 Critic 预测的价值逐步逼近 GAE 算出的真实回报
    value_loss = compute_value_loss(
        values,                      # Critic 当前预测
        old_values,                  # 采样时的旧预测（仅在裁剪模式下用到）
        returns,                     # 拟合目标
        action_mask                  # 屏蔽 pad 位置
    )

    value_loss.backward()            # 反向传播：梯度只流向 Critic
    optimizer_critic.step()          # 更新 Critic 参数

    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  "
          f"value_loss: {value_loss.item():.4f}")
```

**关键设计点：**

1. **新旧概率分离 (Importance Sampling)：** `old_action_log_probs` 来自经验池（采样时刻），`action_log_probs` 来自当前 Actor。这正是 PPO 能够在同一批经验上 **多 epoch 复用** 的理论基础——通过重要性采样修正分布偏移
2. **`zero_grad → backward → step` 三段式：** 每个模型独立完成一次梯度清零 → 反向传播 → 参数更新
3. **梯度互不干扰：** `policy_loss` 只通过 Actor 的计算图，`value_loss` 只通过 Critic 的计算图，所以可以串行更新（详见下文「Actor 与 Critic 是交替训练的吗？」）
4. **TensorBoard 监控：** 两个 loss 都被记录，启动 `tensorboard --logdir ./runs` 即可观察曲线

**log_probs 提取的张量形状变化：**

```python
logits           = actor_model(...).logits                        # (B, L,    V)
logits[:, :-1, :]                                                 # (B, L-1,  V)  去掉最后位
log_probs        = F.log_softmax(...)                             # (B, L-1,  V)
sequences[:, 1:]                                                  # (B, L-1)      右移标签
.unsqueeze(-1)                                                    # (B, L-1,  1)
log_probs_labels = log_probs.gather(dim=-1, index=...)            # (B, L-1,  1)  抽取标签 prob
.squeeze(-1)                                                      # (B, L-1)
[:, -num_actions:]                                                # (B, num_actions)  仅 response
```

- `logits[:, :-1, :]`：第 t 个位置预测第 t+1 个 token
- `sequences[:, 1:]`：右移一位作为标签
- `gather`：从 vocab 维度抽取对应 label 的概率
- `[:, -num_actions:]`：只保留 response 部分

### Actor 与 Critic 是交替训练的吗？

**结论：不是交替训练，而是在每个 `train_step` 中"串行同步更新"**——同一批经验同时用于更新两个模型。

**代码层面的执行顺序：**

每调用一次 `train_step`：

1. 先用 `policy_loss` 反向传播 → `optimizer_actor.step()` 更新 Actor
2. 再用 `value_loss` 反向传播 → `optimizer_critic.step()` 更新 Critic

两个 loss 各自独立反向传播，使用各自的优化器。

**三个层级的循环结构：**

```
for episode in range(episodes):              # 外层：episodes=3 轮
  for prompts in prompts_dataloader:         # 中层：取一批 prompt
      samples = generate_samples(...)        # ① Rollout
      experiences = generate_experiences(...)# ② 算 KL/奖励/优势/回报
      buffer.append(experiences)

      for epoch in range(max_epochs):        # 内层：同一批经验复用 5 轮
          for experience in dataloader:
              train_step(experience, steps)  # ← Actor + Critic 一起更新
```

也就是说：

- 同一批 `experience` 会被**反复使用 `max_epochs=5` 轮**
- 每一轮内，**Actor 和 Critic 都各自更新一次**
- 用完后清空 buffer，重新采样新一批数据

**为什么不需要"交替"：**

PPO 中 Actor 和 Critic 各管各的损失：

| 模型 | 损失函数 | 目标 |
|---|---|---|
| Actor | `policy_loss = -min(ratio·A, clip(ratio)·A)` | 提升高优势动作的概率 |
| Critic | `value_loss = (V − returns)²` | 拟合真实回报 |

二者**梯度互不干涉**（policy_loss 不会传到 Critic 的 `value_head`，value_loss 也不会传到 Actor 的 LM head），所以在同一步串行更新即可，不需要像 GAN 那种「先训 D 再训 G」的交替模式。

**⚠️ 本代码的一个特殊之处：**

注意这一行：

```python
critic_model = Critic(actor_model.base_model).to(device)
```

Critic **直接复用了 Actor 的 `base_model`** 作为特征提取器。这意味着：

- `optimizer_critic = torch.optim.Adam(critic_model.parameters(), ...)` 包含了 `base_model` 的参数
- `optimizer_actor = torch.optim.Adam(actor_model.parameters(), ...)` 也包含了 `base_model` 的参数
- 两个优化器都会修改同一份 base_model 权重 ⚠️

这在工业级实现里通常是**不推荐的做法**（OpenRLHF / TRL 等框架会把 Actor 和 Critic 完全分开，或者只让 `value_head` 独立训练，base_model 冻结或单独管理）。本代码作为教学版本，简化了这一处，理解原理即可，**生产环境建议拆分**。

**一句话总结：**

> Actor 和 Critic 不是交替训练，而是在每个训练步内「串行同步更新」——共用同一批经验、同一个梯度更新周期，但各自的损失和优化器是独立的。

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
