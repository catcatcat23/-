# Softmax‑ST 与 Gumbel‑Softmax‑ST：从零基础到完整推导

> 目标：1) 用尽量简单直白的方式解释 **Softmax‑ST** 与 **Gumbel‑Softmax‑ST** 是什么；2) 严格、不断档地推导为何在 Gumbel 情况下“选到第 k 类的概率”恰好等于 **softmax(z)**（即一次符合真实分类分布的随机抽样）。

---

## 1. 前置知识与符号说明（零基础可读）

- **类别数**：\(K\in\mathbb{N}\)。
- **logits（未归一化得分）**：\(z=(z_1,\dots,z_K)\in\mathbb{R}^K\)。
- **Softmax**（带温度 \(\tau>0\)）：
  \[
  \mathrm{softmax}(z/\tau)_j\;=\;\frac{e^{z_j/\tau}}{\sum_{i=1}^K e^{z_i/\tau}}\quad(j=1,\dots,K).
  \]
  温度 \(\tau\downarrow\) ⇒ 分布更尖锐；\(\tau\uparrow\) ⇒ 更平滑。
- **One‑hot**：第 \(k\) 个位置为 1，其余为 0 的向量，记作 \(\mathrm{one\_hot}(k)\)。
- **Categorical 分布**：\(Y\sim\mathrm{Cat}(p)\) 表示 \(\Pr(Y=k)=p_k\)，\(\sum_k p_k=1\)。
- **Gumbel(0,1) 分布**：
  - 累积分布函数（CDF）：\(F(u)=e^{-e^{-u}}\)。
  - 概率密度函数（PDF）：\(f(u)=e^{-u}\,e^{-e^{-u}}\)。
  - 直觉：在“最大值”问题里非常好用（**Gumbel‑Max 技巧**）。
- **独立性**：\(g_1,\dots,g_K\) 彼此独立（i.i.d.）。
- **Straight‑Through（ST，直通估计器）**：前向用“硬”的（如 one‑hot），反向用“软”的梯度近似来回传（有偏但低方差、实践常用）。

---

## 2. 普通 Softmax 与 Softmax‑ST（不加 Gumbel）

### 2.1 纯软用法（软路由/加权和）
- 定义：\(y^{\text{soft}} = \mathrm{softmax}(z/\tau)\)。
- 用法：把 \(y^{\text{soft}}\) 作为权重做加权和，例如 \(\sum_j y^{\text{soft}}_j f_j(x)\)。全程**可导**。
- 梯度：
  \[
  \frac{\partial y_i^{\text{soft}}}{\partial z_j} = \frac{1}{\tau}\,y_i^{\text{soft}}\big(\mathbf{1}\{i=j\}-y_j^{\text{soft}}\big).
  \]
- 典型场景：**概率校准（温度缩放）**、**混合专家的软组合**。

### 2.2 Softmax‑ST（硬前向 + 软反向）
- 前向：\(y^{\text{hard}}=\mathrm{one\_hot}(\arg\max_j y^{\text{soft}}_j)\)。
- 直通：\(y = (y^{\text{hard}} - y^{\text{soft}})\,\mathrm{detach}() + y^{\text{soft}}\)。
- 效果：前向**单选**（可省算力/便于索引），反向仍用 \(y^{\text{soft}}\) 的梯度。**无探索**（每次都选同一个 argmax）。
- 风险：若训练时用软加权，推理时改成硬单选，会有 **train–test 不一致**。

> 小结（不加 Gumbel）：**可导**，实现简单稳定；但若你需要“随机单选/探索/离散潜变量”，仅靠 softmax 或 ST 还不够。

---

## 3. Gumbel‑Max 与 Gumbel‑Softmax（加入随机噪声）

### 3.1 Gumbel‑Max 技巧（硬采样，精确等价于 Categorical）
- 取 \(g_j\overset{\text{i.i.d.}}{\sim}\mathrm{Gumbel}(0,1)\)，令 \(S_j=z_j+g_j\)。
- 选择：\(\tilde{k}=\arg\max_j S_j\)。
- 结论（核心定理）：
  \[
  \Pr(\tilde{k}=j)\;=\;\frac{e^{z_j}}{\sum_{i=1}^K e^{z_i}}\;=\;\mathrm{softmax}(z)_j.
  \]
  含义：**一次硬选择的分布**，严格等于 **Categorical(softmax(z))**。这就是“符合真实分布的随机抽样”。

> 有温度时：\(\arg\max(z/\tau + g)\) 的选择分布是 \(\mathrm{softmax}(z/\tau)\)。

### 3.2 Gumbel‑Softmax（软样本，可导松弛）
- 定义软样本：
  \[
  y^{\text{g}}\;=\;\mathrm{softmax}\!\Big(\frac{z+g}{\tau}\Big)\quad (g\sim \text{i.i.d. Gumbel}).
  \]
- 性质：\(\tau\downarrow 0\Rightarrow y^{\text{g}}\) 趋近 one‑hot（与 Gumbel‑Max 一致）；\(\tau\uparrow\Rightarrow\) 更平滑。
- **Gumbel‑Softmax‑ST**：前向对 \(y^{\text{g}}\) 取 argmax 变 one‑hot（只跑被选分支），反向仍用 \(y^{\text{g}}\) 的梯度（直通）。
- 好处：**可导随机单选**、天然**探索**（非最大类也有被选概率），与“部署时单选”匹配。

---

## 4. 为什么 \(\Pr(\tilde{k}=j)=\mathrm{softmax}(z)_j\)（完整推导，不跳步）

我们证明在 \(g_i\overset{\text{i.i.d.}}{\sim}\mathrm{Gumbel}(0,1)\) 下：\(\tilde{k}=\arg\max_j(z_j+g_j)\) 满足 \(\Pr(\tilde{k}=j)=e^{z_j}/\sum_i e^{z_i}\)。

### 4.1 连续全概率公式 + 独立性
事件：\(\{\tilde{k}=j\} = \{z_j+g_j\ge z_i+g_i,\ \forall i\}\)。
对 \(g_j\) 条件化（全概率公式，连续型）：
\[
\Pr(\tilde{k}=j)\;=\;\int_{-\infty}^{\infty} \Pr(\tilde{k}=j\mid g_j=x)\, f(x)\,dx,\quad f=\text{Gumbel PDF}.
\]
在 \(g_j=x\) 条件下，\(z_j+x\ge z_i+g_i\iff g_i\le x+z_j-z_i\)。独立性给出
\[
\Pr(\tilde{k}=j\mid g_j=x)\;=\;\prod_{i\ne j} F\!\big(x+z_j-z_i\big),\quad F=\text{Gumbel CDF}.
\]
代回：
\[
\boxed{\ \Pr(\tilde{k}=j)=\int_{-\infty}^{\infty} f(x)\,\prod_{i\ne j}F\!\big(x+z_j-z_i\big)\,dx\ }\tag{★}
\]
> 上式说明了**为何要积分**：我们在对连续随机变量 \(g_j\) 的所有可能取值做“加权求和”（重量是密度 \(f(x)\)）。

### 4.2 把 Gumbel 的 \(F,f\) 代入并化简为 softmax
代入 \(F(u)=e^{-e^{-u}}\)，\(f(u)=e^{-u}e^{-e^{-u}}\)：
\[
\begin{aligned}
\Pr(\tilde{k}=j)
&=\int_{-\infty}^{\infty} e^{-x}e^{-e^{-x}}\prod_{i\ne j} \exp\!\big(-e^{-(x+z_j-z_i)}\big)\,dx\\
&=\int_{-\infty}^{\infty} e^{-x}\,\exp\!\Big(-e^{-x}\big[1+\sum_{i\ne j}e^{-(z_j-z_i)}\big]\Big)\,dx.
\end{aligned}
\]
换元 \(t=e^{-x}\Rightarrow e^{-x}dx=-dt\)，当 \(x\to-\infty,\ t\to\infty\)；\(x\to\infty,\ t\to 0\)：
\[
\Pr(\tilde{k}=j)=\int_{0}^{\infty} \exp\!\Big(-t\,[\,1+\sum_{i\ne j}e^{-(z_j-z_i)}\,]\Big)\,dt.
\]
利用 \(\int_0^{\infty} e^{-\alpha t}dt=1/\alpha\)：
\[
\Pr(\tilde{k}=j)=\frac{1}{1+\sum_{i\ne j}e^{-(z_j-z_i)}}=\frac{1}{\sum_{i=1}^K e^{z_i-z_j}}=\frac{e^{z_j}}{\sum_{i=1}^K e^{z_i}}=\mathrm{softmax}(z)_j.\quad\blacksquare
\]
这说明 **Gumbel‑Max 的硬选择**，**精确**地等价于从 \(\mathrm{Cat}(\mathrm{softmax}(z))\) 抽一次样本。

> 若使用温度 \(\tau\)（将 \(z\) 换成 \(z/\tau\)），则 \(\Pr(\tilde{k}=j)=\mathrm{softmax}(z/\tau)_j\)。

---

## 5. Softmax‑ST 与 Gumbel‑Softmax‑ST 的差别（目标与用法）

- **Softmax‑ST（无 Gumbel）**：
  - 前向：确定性 **单选 argmax**；无随机性、**无探索**。
  - 反向：用软的 \(\mathrm{softmax}(z/\tau)\) 传播梯度（有偏直通）。
  - 适合：你只想**确定性单选**且要省算力，但不追求探索（冷门分支不易被训练）。

- **Gumbel‑Softmax‑ST（有 Gumbel）**：
  - 前向：\(\arg\max(z+g)\)（或 \(\arg\max\,\mathrm{softmax}((z+g)/\tau)\)）**随机单选**，\(\Pr\) 正是 \(\mathrm{softmax}\)。
  - 反向：用 \(\mathrm{softmax}((z+g)/\tau)\) 的梯度（直通）。
  - 适合：**要单选 + 要探索**（每个分支有非零被选概率）、**离散潜变量**、与“部署时单选”**一致**。

> 目标差异可用“期望与损失的次序”理解：
> - 软路由优化的是 \(L(\sum_j y_j f_j)\)（**混合输出**）。
> - 随机单选优化的是 \(\mathbb{E}[L(f_{\text{单选}})]\)（**抽样单选的期望损失**）。一般 \(L(\mathbb{E}[\cdot])\ne \mathbb{E}[L(\cdot)]\)。

---

## 6. 代码模板（PyTorch 伪代码）

**A. Softmax‑ST（无 Gumbel）**
```python
import torch, torch.nn.functional as F

def softmax_st(logits, tau=1.0):
    y_soft = F.softmax(logits / tau, dim=-1)
    y_hard = torch.zeros_like(y_soft)
    y_hard.scatter_(1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
    # forward 用硬的，backward 用软梯度
    y = (y_hard - y_soft).detach() + y_soft
    return y  # one-hot (forward), soft-grad (backward)
```

**B. Gumbel‑Softmax‑ST（有 Gumbel）**
```python
import torch, torch.nn.functional as F

def sample_gumbel(shape, eps=1e-9):
    u = torch.rand(shape)
    return -torch.log(-torch.log(u + eps) + eps)  # Gumbel(0,1)

def gumbel_softmax_st(logits, tau=1.0):
    g = sample_gumbel(logits.size()).to(logits.device)
    y_soft = F.softmax((logits + g) / tau, dim=-1)  # 随机、可导
    y_hard = torch.zeros_like(y_soft)
    y_hard.scatter_(1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
    y = (y_hard - y_soft).detach() + y_soft  # ST 直通
    return y
```

---

## 7. 常见易混点与答疑

1) **“不加 Gumbel 我也能 ST，为什么还要 Gumbel？”**  
ST 解决“硬选择如何回传梯度到 **logits**”的问题；但**不产生探索**，未被选的分支拿不到样本与梯度。Gumbel 让每个分支以 \(\mathrm{softmax}(z)\) 的概率被选中，天然探索，且与“部署时单选”匹配。

2) **“Gumbel‑Softmax 的软样本 \(\mathbb{E}[y^{\text{g}}]\) 是否等于 \(\mathrm{softmax}(z)\)？”**  
严格等价的是 **硬选择**的分布：\(\Pr(\arg\max(z+g)=k)=\mathrm{softmax}(z)_k\)。软样本的期望一般不等于 \(\mathrm{softmax}(z)\)，但 \(\tau\to 0\) 时 \(y^{\text{g}}\) 逼近 one‑hot，行为与硬选择一致。

3) **“为什么推导里要积分？”**  
因为我们对连续随机变量（如 \(g_j\) 或最大值“高度” \(t\)）做**全概率加总**；连续“加总”=**积分**。详见 (★) 式前后的说明。

---

## 8. 何时用哪个（选型清单）

- **只做校准/软融合**（不关心采样/探索）→ 用 **Softmax** 或 **Softmax‑ST**。
- **必须单选 + 需要探索/离散潜变量/与部署一致** → 用 **Gumbel‑Softmax‑ST**（配温度退火）。
- **想省算力**（只算被选分支）→ 任选 **ST**；若还想探索 → **Gumbel‑ST** 更合适。

---

## 9. 一页小结

- **Softmax‑ST**：前向硬单选，反向软梯度；确定性、无探索；训练目标更像“单选的近似”。
- **Gumbel‑Softmax‑ST**：前向随机单选（\(\Pr=\mathrm{softmax}\)），反向软梯度；可导、可探索、与部署一致。
- **关键定理**：\(\Pr\big(\arg\max_j (z_j+g_j)=k\big)=\mathrm{softmax}(z)_k\)，其推导基于：
  - 连续型**全概率公式**与**独立性**；
  - **次序统计**最大者位置的通式；
  - Gumbel 的 \(F,f\) 代入与**变量代换积分**；
  - 最终化简为 **softmax**。

> 一句话：**Gumbel‑Max 给出“精准的分类抽样”，Gumbel‑Softmax 给出“可导的连续松弛”，ST 让我们“前向硬、反向软”。** 选择哪个，取决于你需要“混合”还是“单选”，是否需要“探索”，以及是否要与部署时的行为一致。

