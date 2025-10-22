# STS（Simplex Temperature Scaling）方法学习总结与复习提纲

> 面向“从零到一”的复习文档：先把**基础数学**讲清楚（符号、定义、定理、推导），再一步步搭建到 **STS 的完整方法**（模型、训练、推理、直觉、对比、易错点）。每段都标注**所用数学工具**与**前置知识**，可按小标题逐段复习。

---

## 0. 统一符号（全篇通用）

- **类别数**：$K\in\mathbb{N}$；**标签空间**：$\mathcal{Y}=\{1,\dots,K\}$。
- **输入/标签**：$x\in\mathcal{X}$，$y\in\mathcal{Y}$。
- **概率单纯形**：$\displaystyle \Delta_{K-1}=\{\pi\in\mathbb{R}^K_{\ge 0} : \sum_{k=1}^K \pi_k = 1\}$。
- **概率向量随机变量**：$\pi\in\Delta_{K-1}$。
- **预训练模型**：特征提取器 $\hat f$，分类头 logits $\hat g(x)=(g_1,\dots,g_K)$。
- **位置参数**：$\alpha_k(x)=e^{g_k(x)}>0$，记 $\alpha(x)=(\alpha_1,\dots,\alpha_K)$。
- **温度支路**：$\lambda(x)=\mathrm{softplus}((h\circ \hat f)(x))>0$。

> 记忆卡：$\alpha$ 决定 **类概率的相对权重**；$\lambda$ 决定 **单纯形上分布的“尖/钝”**。

---

## 1. 基础数学与直觉

### 1.1 Gamma **函数**与“阶乘的推广”

- 定义（欧拉积分）：$\displaystyle \Gamma(s)=\int_0^\infty t^{s-1}e^{-t}\,dt\quad (s>0)$。
- 性质：$\Gamma(s+1)=s\,\Gamma(s)$，故 $\Gamma(n)=(n-1)!$（把阶乘平滑推广到非整数）。
- 前置知识：积分、分部积分、指数函数。

### 1.2 泊松过程 → 指数/Gamma **分布**

- 齐次泊松过程（强度 $\lambda$）：固定时长内计数 $N(t)\sim\mathrm{Poisson}(\lambda t)$。
- **第一次到达时间**：$X_1\sim\mathrm{Exp}(\lambda)$，用 $\mathbb{P}(X_1>t)=\mathbb{P}(N(t)=0)=e^{-\lambda t}$ 得到。
- **第 $k$ 次到达时间**：$T_k=\sum_{i=1}^k X_i\sim\mathrm{Gamma}(k,\lambda)$（Erlang）。
- 前置知识：生存函数/分布函数/密度函数、卷积、拉普拉斯/矩母函数。

### 1.3 Dirichlet：单纯形上的“比例分布”

- 定义：$w\sim\mathrm{Dir}(\alpha_1,\dots,\alpha_K)$ 的密度 $\propto \prod_k w_k^{\alpha_k-1}$。
- 构造（关键）：**独立同率 Gamma 归一化**：若 $g_k\sim\mathrm{Gamma}(\alpha_k,\beta)$ 独立，$w_k=g_k/\sum_i g_i \Rightarrow w\sim\mathrm{Dir}(\alpha)$。
- 直觉：$\alpha_k$ 是“伪计数/用时块长”的强度；$\alpha_0=\sum_k\alpha_k$ 越大，比例越稳定。
- 前置知识：变量代换、雅可比行列式（$|\det|=S^{K-1}$）。

### 1.4 Gumbel-max、指数竞赛与 softmax

- **Gumbel-max**：对 $R_k=\ln\alpha_k+G_k$（$G_k\sim\mathrm{Gumbel}(0,1)$ 独立），有 $\mathbb{P}(\arg\max R=k)=\alpha_k/\sum_i\alpha_i$。
- **指数竞赛**（等价）：取 $T_k\sim\mathrm{Exp}(\alpha_k)$，则 $\mathbb{P}(\arg\min T=k)=\alpha_k/\sum_i\alpha_i$。
- 结论：$\mathrm{softmax}(g)_k=\alpha_k/\sum_i\alpha_i$。
- 前置知识：独立性、积分、换元；Gumbel 与指数的联系（$-\ln\mathrm{Exp}(1)\sim\mathrm{Gumbel}$）。

### 1.5 Concrete（Gumbel-Softmax）分布

- 采样式：$\displaystyle \pi=\mathrm{softmax}\!\Big(\frac{\ln\alpha + G}{\lambda}\Big)\in\Delta_{K-1}$。
- 几何直觉：$\lambda\downarrow$ 则质量向顶点集中（更“尖”、自信）；$\lambda\uparrow$ 则向中心集中（更“钝”、不自信）。
- 用途：给 $\pi$ 在单纯形上的**连续概率模型**，可微近似“抽 one-hot”。

---

## 2. STS 方法：建模思想与两层结构

### 2.1 两层随机模型（核心定义）

1. **决策层**：$p(y=k\mid \pi)=\mathbf{1}\{\arg\max_i \pi_i=k\}$（赢家通吃）。
2. **单纯形层**：$p(\pi\mid x)=\mathrm{Concrete}(\pi\mid \alpha(x),\lambda(x))$。

> 读法：**类别概率**来自 $\pi$ 的赢家；$\pi$ 的形状由 $(\alpha,\lambda)$ 决定，其中 $\alpha$ 由 logits 指数化（位置），$\lambda$ 由温度支路输出（形状/扩散）。

### 2.2 关键定理：$p(y\mid x)$ **与 $\lambda$ 无关**

- 从 $p(y=k\mid x)=\int \mathbf{1}\{\arg\max\pi=k\}\,p(\pi\mid x)\,d\pi$ 出发。
- 用 **Gumbel-max** 化简：$\arg\max \pi = \arg\max (\ln\alpha + G)$（除以正 $\lambda$ 不改 $\arg\max$）。
- 得 $p(y=k\mid x)=\mathbb{P}(\arg\max(\ln\alpha+G)=k)=\alpha_k/\sum_i\alpha_i=\mathrm{softmax}(g)_k$。
- 结论：**分类分布**由 $\alpha$（即 logits）唯一决定，**不依赖**温度 $\lambda$。

### 2.3 推论：Step‑1 的交叉熵最优性与准确率保持

- 用 $\alpha_k=e^{g_k}$ 得 $p(y\mid x)=\mathrm{softmax}(g)$。最大似然 $\Leftrightarrow$ 最小化**交叉熵**（标准 softmax CE）。
- Step‑2 只训练 $\lambda$，不会改变 $p(y\mid x)$ 与 $\arg\max$，因此**准确率不变**，只影响**不确定性刻度**。

---

## 3. 训练：Multi‑Mixup 与温度分支 NLL

### 3.1 Multi‑Mixup（为什么 & 怎么做）

- 目标：给温度支路提供“**靠顶点**/低不确定”与“**靠中心**/高不确定”两类样本。
- 做法：**按类各取一条**样本，再从 $w\sim\mathrm{Dir}(\beta\mathbf{1}_K)$ 抽权重，线性混合
  $\tilde x=\sum_k w_k x^{(k)}$，$\tilde\pi=\sum_k w_k y^{(k)}=w$。
- 选 Dirichlet 的理由：支持集是单纯形；$\beta<1$ 靠顶点，$\beta>1$ 靠中心；$\beta=1$ **单纯形均匀**（与维度无关）；$K=2$ 退化成 Beta（与经典 Mixup 一致）。

### 3.2 温度支路的目标函数（Concrete‑NLL，逐项展开）

对合成集 $\{(\tilde x_m,\tilde\pi_m)\}_{m=1}^M$ 最小化
$$
L(\theta_\lambda)=-\frac{1}{M}\sum_m \ln \mathrm{Concrete}\!\big(\tilde\pi_m \,\big|\, \alpha(\tilde x_m),\, \lambda(\tilde x_m)\big).
$$
把 Concrete 密度分解并逐项取 $-\ln$：

- $-(K-1)\ln\lambda$（温度项）；
- $-\sum_k\ln\alpha_k$（位置常数项；与 $\lambda$ 无关）；
- $+\sum_k (\lambda+1)\ln \tilde\pi_k$（幂指数项）；
- $+K\ln\!\big(\sum_i \alpha_i\,\tilde\pi_i^{-\lambda}\big)$（归一化项）。

> 训练时 **只反传到温度支路 $h$**，$\hat f,\hat g$ 冻结。

### 3.3 置信度与 MC 估计

- 传统置信度 $\max_k p(y=k\mid x)$ 与 $\lambda$ 无关 ⇒ 无法校准。
- STS 定义：$\displaystyle \mathrm{conf}(x)=\max_k \big(\mathbb{E}[\pi\mid x]\big)_k$。
- 近似：采 $G^{(j)}$（Gumbel），$\hat\pi^{(j)}=\mathrm{softmax}((\ln\alpha+G^{(j)})/\lambda)$，以 $\frac{1}{p}\sum_j \hat\pi^{(j)}$ 近似 $\mathbb{E}[\pi]$。

### 3.4 不确定性分解（可选）

- Aleatoric（数据固有）：$\mathbb{E}\!\left[-\sum_k \pi_k\ln\pi_k\right]$。
- Epistemic（模型）：$\mathbb{E}\!\left[-\ln p(\pi\mid x)\right]$。
- 皆可用同一批 $\hat\pi^{(j)}$ 做 MC 估计。

---

## 4. 实践流程与伪代码（复习用）

### 4.1 训练两步法

**Step‑1 分类头（冻结为 $\hat g$）**：

1. 训练 $\hat f, \hat g$ 最小化交叉熵；
2. 得到 $\alpha(x)=e^{\hat g(x)}$。

**Step‑2 温度支路（只训 $h$）**：

1. 用分班 Mini‑batch 构造 Multi‑Mixup：每类取 $R$ 个样本，做 $S$ 轮；
2. 每次从 $\mathrm{Dir}(\beta\mathbf{1}_K)$ 抽 $w$，生成 $(\tilde x,\tilde\pi)$；
3. 前向得 $\lambda(\tilde x)$，计算 Concrete‑NLL 并反传更新 $h$。

### 4.2 推理（保持准确率、输出校准置信）

- 预测类别：$\arg\max_k \hat g_k(x)$（与训练前一致）。
- 置信度：按 §3.3 的 MC 近似输出 $\max_k \widehat{\mathbb{E}[\pi_k]}$。

### 4.3 关键超参与建议

- $\beta$：控制合成标签分布（0.2–2.0 网格选；$\beta<1$ 稀疏、$>1$ 中心、$=1$ 均匀）。
- $R,S$：决定每步合成样本数 $M=RS$ 与多样性；$R$ 大通常更稳。
- MC 样本数 $p$：10–100 视预算折中（越大越稳）。

---

## 5. 概念对照与易错点

### 5.1 与 PTS/AdaTS 对比

- 三者温度均为样本依赖；但 **STS 显式在单纯形上建模 $p(\pi\mid x)$**。
- **准确率**：STS 的 $p(y\mid x)$ 与 $\lambda$ 无关 ⇒ **保持**；PTS/AdaTS 对 logits 乘正温度也不改 $\arg\max$。
- **不确定性**：STS 可分解 aleatoric/epistemic；PTS/AdaTS 难以自然分解。

### 5.2 Dirichlet + 只训温度为何容易“极端低置信”

- 在固定 logits 下，仅靠“温度式缩放”的 Dirichlet 难以匹配内部合成标签形状 ⇒ 优化把温度推大，整体概率被压扁（underconfidence）。
- 若坚持 Dirichlet，应 **联合**调“位置 + 温度/浓度”，并约束不破坏 Step‑1 的 CE 最优性。

### 5.3 常见混淆

- **softmax ≠ Gamma(形状 > 1)**：softmax 与“**指数竞赛** / Gumbel‑max”等价（形状 = 1）。
- **Multi‑Mixup 的 $w$**：不是“数据天然如此”，而是**设计为 Dirichlet**，以覆盖单纯形、可控多样性、与 Mixup 兼容。

---

## 6. 一页速记（Cheat Sheet）

- $\alpha=e^{g}$，$\lambda>0$。
- $p(y=k\mid x)=\alpha_k/\sum_i\alpha_i = \mathrm{softmax}_k(g)$（与 $\lambda$ 无关）。
- Concrete 采样：$\pi=\mathrm{softmax}((\ln\alpha+G)/\lambda)$。
- 置信度：$\max_k \mathbb{E}[\pi_k]\approx \max_k \frac{1}{p}\sum_j \hat\pi^{(j)}_k$。
- Multi‑Mixup：$w\sim\mathrm{Dir}(\beta\mathbf{1})$，$\tilde x=\sum_k w_k x^{(k)}$，$\tilde\pi=w$。
- NLL（温度支路）：$-(K-1)\ln\lambda -\sum_k\ln\alpha_k +\sum_k(\lambda+1)\ln\tilde\pi_k + K\ln\sum_i \alpha_i\,\tilde\pi_i^{-\lambda}$。

---

## 7. 复习清单（按题自检）

1. 能否用 **Gumbel‑max**（或指数竞赛）推导 $\mathrm{softmax}$？  
2. 能否从 **Concrete 密度**逐项推到温度支路 NLL？  
3. 为什么 $p(y\mid x)$ 与 $\lambda$ 无关、准确率保持？  
4. 为什么 Multi‑Mixup 用 **Dir($\beta\mathbf{1}$)** 合理？$\beta$ 如何影响合成标签？  
5. 置信度为何换成 $\max_k\mathbb{E}[\pi_k]$？如何做 MC 估计？  
6. Aleatoric / Epistemic 如何由 $\pi$ 与 $p(\pi\mid x)$ 定义并估计？  

---

## 8. 前置知识速览（用到哪些数学）

- 微积分：分部积分、变量代换、雅可比、极坐标。
- 概率：泊松过程、指数与 Gamma、卷积、MGF/拉氏、指示函数期望 = 事件概率。
- 随机优化：最大似然、交叉熵、Monte Carlo 估计、Softplus 与可微编程。

> 建议：先按 §1 过一遍基础；再按 §2–§4 跑通一次“从定义到训练/推理”的链条；最后用 §6–§7 做速记与自测。准备考试/汇报时只需带着本页即可。
