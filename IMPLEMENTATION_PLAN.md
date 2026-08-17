# 结构映射可视化 · 设计与实施计划

目标：参照 [bbycroft.net/llm](https://bbycroft.net/llm) 的核心手法——**几何由张量结构决定**。
节点的宽、高、叠片深度全部从真实维度推出来，维度数字标在边上；不加任何新的颜色、渐变、阴影等视觉样式。

现状问题：`js/walkthrough.js` 里所有节点固定 150×70，`matrixCells()` 画的 3×4 格子是纯装饰，
与真实 shape 无关。本次重构删掉装饰，让几何承载信息。

## 设计规范

### 1. 空间语义（核心约定）

| 视觉轴 | 结构含义 | 标度 |
|---|---|---|
| 横向宽度 | 特征/列维（H、latent、vocab、dOut…） | log₂ |
| 纵向高度 | 行/token 维（B×T、B×S、dIn…） | log₂ |
| 叠片深度（2.5D 偏移副本） | 重复轴：heads / experts / 层数 | 份数标注 ×N |
| 盒内网格纹理 | 「这是矩阵」的提示（共享 pattern，非逐 cell DOM） | 固定 12×10px |

标度函数（两轴统一用，参数写死避免执行方自由发挥）：

```text
px(n, base, per, cap) = min(cap, base + per × log₂(max(1, n)))
宽（列维）:  px(n, 26, 9.5, 320)
高（行维）:  px(n, 20, 7.5, 240)
最小宽 56（保证文字可读）；chip 兜底 120×44
```

校准点：H=7,168 → 宽≈148px；vocab=163,840 → ≈190px；latent=3,584 → ≈138px；
kvLatent 576 → ≈113px。Prefill 激活 B×T=8×19.2K → 高≈149px；Decode 激活 B=8 → ≈42px；
KV cache B×S=512K → ≈162px。**Prefill/Decode 切换后激活高度肉眼可辨**，这是本设计的验收要点。

### 2. 几何解析 `nodeGeometry(item, ctx)`

新增一个与 `resolveShape` 平行的**数值**解析器：`{B}/{T}/{S}` 代入原始整数（不做 19K 格式化），
去掉千分位逗号，取 `[...]` 括号组，按 `,` 和 `×` 拆分，每段内数字相乘。分类规则按优先级：

1. `kind === "weight"` 或 `"module"` 且带 `dIn`/`dOut` → **矩阵**：rows=dIn，cols=dOut。
2. 括号内 2 维 `[a, b]` → rows=a, cols=b（如 logits `[B, 163840]`）。
3. 括号内 3 维 `[a, b, c]` → rows=a×b, cols=c（如 hidden `[B, T, H]`）。
4. 括号内 4 维 `[a, h, t, s]` → stack=h（heads），rows=a×t, cols=s（如 KDA state `[B, 96, 128, 128]`）。
5. 无括号但结尾有数字（如 `block states × 7,168`）→ rows=1 的横条，cols=末尾数。
6. 都不中 → chip 兜底（rope、short_conv 等标注性节点）。

### 3. 重复轴：叠片渲染

- face 矩形后面画最多 3 层偏移副本，偏移 (5,−5)、(10,−10)，透明度递减（0.5 / 0.25），
  右上角标 `×N`（N 用紧凑格式：96、896、64…）。
- stack 来源优先级：图元数据 `extra.stack` > 4 维 shape 的 dims[1] > 无。
- `js/model-graphs.js` 只加元数据不改结构，需要补 `extra.stack`（及说明用 `extra.stackNote`）的节点：
  - K3：`gated_mla` ×96；`routed_experts` ×896（note "16 active"）；`expert_compute` ×896；`shared_experts` ×2（`kda_state` 由 4 维自动推出 ×96）
  - GLM-5.2：`sparse_attention` ×64；`routed_experts`/`expert_compute` ×256（note "8 active"）（`index_scores` 4 维自动 ×32）
  - K2.5：`routed_experts`/`expert_compute` ×384（note "8 active"）（`mla_attn` 4 维自动 ×64）
  - M3：`sparse_gqa` ×64；`kv_cache` ×4（KV heads）；`indexer` ×4；`routed_experts`/`expert_compute` ×128（note "4 active"）
- **Decoder Layer 外框**同样叠 2 层偏移副本，表示 ×93 层（GLM ×78、K2.5 ×61、M3 ×60），
  现有 `layerNote`（"69 KDA + 24 MLA" 等）保留在框标题里。

### 4. MoE 专家格

`routed_experts` / `expert_compute` 的矩形内部画粗粒度专家格：
cell 数 = min(experts, 128)，约 16 列排布；被激活的 Top-K 个 cell 用该 kind 颜色实心（opacity 1），
其余 0.15。让「896 里亮 16 个」直接可见。DOM 上限 128 个 rect，不许逐专家画 896 个。

### 5. 维度刻度与文字

- 列维标签：盒子底边下方居中（y+h+10），等宽 9px。静态维度用千分位原值（"7,168"），
  含 token 的用紧凑格式（"153K"）。
- 行维标签：盒子左边，rotate(−90) 垂直居中（x−6）。同上格式规则。
- 节点名：盒子上方居中（y−8），沿用现有字体样式。
- 序号：保留右上角。
- **删除** `matrixCells()` 装饰格子，换成共享 `<pattern>` 网格叠层（低透明度）。

### 6. 布局改造 `computeLayout`

- 行高改为变量：`rowHeight = max(该行节点高) + 64`（28 名字留白 + 36 边线间隙）；组间 +58 保留。
- 行内横向：`total = Σwidth + 48×(n−1)`，起点 `axis − total/2`，逐个排。
- `positions` 里每个节点存自己的 width/height（边线、外框已按 box.width/height 取值，无需改）。
- 残差绕行边的触发阈值（现为 `endY−startY > 150`）在变高行下需要复核，必要时改为「跨了≥2 行」判定。

### 7. 图例（index.html）

`.stage-legend` 追加一条说明（保留现有 kind 色块）：
`宽 ∝ log₂列维 · 高 ∝ log₂行/token · 叠片×N = heads/experts/层 · 网格 = 矩阵`

### 8. 边界（不许做的事）

- 不加新配色/渐变/阴影/动画；不引入 WebGL 或第三方库，保持纯 SVG。
- 不改 `js/roofline.js`、`js/app.js`（`render(state)` 签名与入参不变，ctx 已含 B/T/S）。
- `BallparkWalkthrough` 导出 API 不变：`render, visibleNodes, resolveShape, upstreamIds, topologicalOrder`。
- 交互不变：点击/键盘选中、autoscroll、zoom fit、播放高亮 class
  （`is-selected / is-upstream / is-done / is-pending / is-dimmed / is-hot`）。

## 验收标准

1. Embedding / LM head 是全图最宽的权重（163,840 列）；latent 投影明显窄于 H 宽 GEMM；
   Decode 下 KV cache 明显高（S 行）而激活明显矮（只有 B 行）。
2. Prefill ↔ Decode 切换只改激活高度，布局不塌、边不断。
3. K3 的 KDA/MLA 分支都能渲染：KDA state 是 128×128 方块 + ×96 叠片。
4. 四个模型 × 两种模式全部渲染，播放路径可走完，console 无报错。
5. 「适应」缩放仍能按容器宽收纳全图。

## Stages

### Stage 1: 几何解析与布局
**Goal**: `nodeGeometry()` + 变尺寸 `computeLayout()`，节点按规范出尺寸
**Success Criteria**: 校准点尺寸与规范一致（误差 ±2px）；四模型布局不重叠
**Tests**: `node --check js/walkthrough.js`；浏览器目测四模型
**Status**: Not Started

### Stage 2: 节点渲染
**Goal**: 矩阵网格 pattern、叠片、专家格、维度刻度、删装饰格子
**Success Criteria**: 验收标准 1–3
**Tests**: 浏览器截图对照校准点
**Status**: Not Started

### Stage 3: 图元数据与图例
**Goal**: model-graphs.js 补 `extra.stack/stackNote`；index.html 图例一行
**Success Criteria**: 叠片 ×N 与架构数一致（96/896/64/…）
**Tests**: 逐模型核对
**Status**: Not Started

### Stage 4: 浏览器验证
**Goal**: 四模型 × Prefill/Decode × K3 双分支全过一遍 + 播放路径录屏
**Success Criteria**: 验收标准 4–5，console 干净
**Tests**: `python3 -m http.server 8000` 手测
**Status**: Not Started
