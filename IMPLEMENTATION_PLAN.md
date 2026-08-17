# 结构映射可视化 · 设计与实施计划

目标：参照 [bbycroft.net/llm](https://bbycroft.net/llm) 重构架构图——**几何由张量结构决定，
整体呈现让"模型形状"可见**。不加新配色/渐变/阴影，保持现有纸面风格。

现状问题：`js/walkthrough.js` 所有节点固定 150×70，`matrixCells()` 的 3×4 格子是纯装饰；
整个模型只画一个代表层 + "×93 层"文字，模型的整体形状（巨大的 embedding、93 层薄堆、
巨大的 lm_head）不可见。

## 对照 bbycroft：采纳什么、不采纳什么

| bbycroft 做法 | 本项目 | 理由 |
|---|---|---|
| 张量尺寸 = 真实维度 | ✅ 采纳（log₂ 标度） | 核心诉求 |
| 全模型物理渲染（每层都画） | ✅ 变体：代表层展开 + 其余层压缩片堆 | 93 层全展开 DOM 爆炸 |
| 残差流作为中轴主线 | ✅ 采纳 | 结构即布局 |
| 章节导航 + 相机飞行 | ✅ 变体：组标签可点击滚动定位 | 已有 autoscroll 基础 |
| 缩放为核心交互 | ✅ 采纳：滚轮缩放 + 拖拽平移 | 高画布必须能导航 |
| 计算流动画 | ✅ 变体：播放时激活边虚线流动 | 方向信息，非装饰 |
| 维度标注在张量边上 | ✅ 采纳 | 核心诉求 |
| 逐 cell 权重渲染 / LOD | ❌ 用共享网格 pattern 替代 | SVG DOM 上限 |
| WebGL / 3D 相机 | ❌ 保持 SVG 2.5D | 零依赖原则 |
| 暗色发光风格 | ❌ 保持纸面风格 | 明确不做样式花样 |

## 设计规范

### 1. 空间语义（核心约定）

| 视觉轴 | 结构含义 | 标度 |
|---|---|---|
| 横向宽度 | 特征/列维（H、latent、vocab、dOut…） | log₂ |
| 纵向高度 | 行/token 维（B×T、B×S、dIn…） | log₂ |
| 叠片深度（2.5D 偏移副本） | 重复轴：heads / experts | ×N 标注 |
| 纵向片堆 | 层重复（93 层按真实种类交错） | 每层一片 |
| 盒内网格纹理 | 「这是矩阵」提示（共享 pattern） | 固定 12×10px |

标度函数（参数写死，不留发挥空间）：

```text
px(n, base, per, cap) = min(cap, base + per × log₂(max(1, n)))
宽（列维）:  px(n, 26, 9.5, 320)
高（行维）:  px(n, 20, 7.5, 240)
最小宽 56；chip 兜底 120×44
```

校准点：H=7,168 → 宽≈148px；vocab=163,840 → ≈190px；latent=3,584 → ≈138px；
kvLatent 576 → ≈113px。Prefill 激活 B×T=8×19.2K → 高≈149px；Decode 激活 B=8 → ≈42px；
KV cache B×S=512K → ≈162px。**Prefill/Decode 切换后激活高度肉眼可辨**。

### 2. 几何解析 `nodeGeometry(item, ctx)`

与 `resolveShape` 平行的**数值**解析器：`{B}/{T}/{S}` 代入原始整数，去千分位逗号，
取 `[...]` 括号组，按 `,`/`×` 拆分，段内数字相乘。分类优先级：

1. `weight`/`module` 且带 `dIn`/`dOut` → 矩阵：rows=dIn，cols=dOut。
2. 2 维 `[a, b]` → rows=a, cols=b。
3. 3 维 `[a, b, c]` → rows=a×b, cols=c。
4. 4 维 `[a, h, t, s]` → stack=h, rows=a×t, cols=s（如 KDA state `[B, 96, 128, 128]`）。
5. 无括号但结尾有数字 → rows=1 横条，cols=末尾数。
6. 兜底 chip（rope、short_conv 等标注性节点）。

### 3. 全模型柱（bbycroft 的"模型形状"）

画布纵向依次：Embedding（真实比例）→ **代表层展开**（现有 DAG 细节，框标"Layer 1 · 代表层"）
→ **压缩层堆**（其余 N−1 层，每层一个薄片）→ Output head（真实比例）。

压缩层堆规则：

- 片高 `clamp(2, 240 / layers, 6)` px，宽 = 主干 H 的 px 宽；水平居中于中轴。
- 每片按**真实层种类与顺序**着色区分（复用现有 CSS 变量色，不新增色值）：
  - K3：KDA 片与 MLA 片按 `(3 KDA + 1 MLA) × 23 + 1 MLA` 交错（从 dims 的
    kdaLayers=69 / mlaLayers=24 推导，禁止硬编码 93）。
  - GLM-5.2：前 3 dense 片 + 75 MoE 片；K2.5：1 + 60；M3：前 3 dense + 57 sparse。
- 堆旁标注计数（如"其余 92 层 · 69 KDA + 24 MLA"）。
- 点击堆中任意片 → 滚动到代表层（片本身不展开）。

### 4. 残差主轴

残差流激活钉在中轴（x 居中），旁路分支左右侧挂：

- `model-graphs.js` 给残差链激活加 `extra.spine: true`：
  `hidden_in、attn_residual/attention_output、block_residual/block_output、final_norm、logits`
  （四个模型同规则）。
- 布局：spine 节点强制中轴对齐；同行其余节点在两侧分布。
- 残差绕行边（现 `endY−startY > 150` 触发侧绕）改为「跨 ≥2 行」判定，在变高行下复核。

### 5. 重复轴：叠片渲染

- face 后画最多 3 层偏移副本，偏移 (5,−5)、(10,−10)，opacity 0.5 / 0.25，右上角标 `×N`。
- stack 来源优先级：`extra.stack` > 4 维 shape 的 dims[1] > 无。
- `model-graphs.js` 需补 `extra.stack`（说明用 `extra.stackNote`）：
  - K3：`gated_mla` ×96；`routed_experts`/`expert_compute` ×896（"16 active"）；`shared_experts` ×2
  - GLM-5.2：`sparse_attention` ×64；`routed_experts`/`expert_compute` ×256（"8 active"）
  - K2.5：`routed_experts`/`expert_compute` ×384（"8 active"）
  - M3：`sparse_gqa` ×64；`kv_cache` ×4；`indexer` ×4；`routed_experts`/`expert_compute` ×128（"4 active"）

### 6. MoE 专家格

`routed_experts`/`expert_compute` 矩形内画粗粒度专家格：cell 数 = min(experts, 128)，
约 16 列；Top-K 个 cell 实心（opacity 1），其余 0.15。DOM 上限 128 个 rect。

### 7. 维度刻度与文字

- 列维标签：底边下方居中（y+h+10），等宽 9px；静态维度千分位原值（"7,168"），
  含 token 的用紧凑格式（"153K"）。
- 行维标签：左边 rotate(−90) 垂直居中（x−6）。
- 节点名：盒上方居中（y−8）；序号保留右上角。
- **删除** `matrixCells()`，换共享 `<pattern>` 低透明度网格。

### 8. 导航与播放

- **滚轮缩放**（以光标为中心）+ **拖拽平移**，双击 = 适应宽度；现有 ± / 适应按钮保留，
  内部走同一 zoom 状态（walkthrough 通过 `onZoom(next)` 回调把状态交还 app.js）。
- **章节锚点**：组标签（EMBEDDING / ATTENTION / MOE FFN / OUTPUT HEAD）可点击，
  平滑滚动到该组（复用 autoscroll 的坐标换算）。
- **播放流动**：播放态下 `.walkthrough-edge.is-active` 加 stroke-dasharray 流动动画
  （一条 CSS 规则）；非播放态静止。

### 9. 布局改造 `computeLayout`

- 行高变量化：`rowHeight = max(该行节点高) + 64`；组间 +58 保留。
- 行内横向：spine 居中，其余 `Σwidth + 48 gap` 两侧排布。
- `positions` 存每节点实际 width/height（边、框已按 box 取值）。

### 10. 允许改动的文件与边界

| 文件 | 允许 | 禁止 |
|---|---|---|
| `js/walkthrough.js` | 全部重写 | 导出 API 改名（`render, visibleNodes, resolveShape, upstreamIds, topologicalOrder` 保留） |
| `js/model-graphs.js` | 加 `extra.stack/stackNote/spine` 元数据 | 改节点结构、依赖关系、costId |
| `js/app.js` | 交互绑定（wheel/pan/章节/onZoom）、向 render 传 `dims` 等结构数据 | 改成本计算调用、控件读数逻辑 |
| `index.html` | 图例一行 + 必要的 CSS 类 | 新增色值、改主题 |
| `js/roofline.js` | 仅加只读导出（如需暴露既有数据） | 改任何公式/数值路径——吞吐数字有 GPUStack 实测锚点（±6%）校验，改坏即失真 |

交互不变：点击/键盘选中、播放高亮 class（`is-selected / is-upstream / is-done / is-pending / is-dimmed / is-hot`）。

### 11. 图例（index.html）

`.stage-legend` 追加：`宽 ∝ log₂列维 · 高 ∝ log₂行/token · 叠片×N = heads/experts · 片堆 = 层 · 网格 = 矩阵`。

## 验收标准

1. 缩小到全景时能看出**模型形状**：宽 embedding → 细高 93 层片堆 → 宽 lm_head；
   K3 片堆中 KDA/MLA 交错可辨。
2. Embedding / LM head 是最宽权重；Decode 下 KV cache 高（S 行）、激活矮（B 行）；
   Prefill ↔ Decode 切换只改激活高度，布局不塌。
3. 残差流在中轴上下贯通，attention/MoE 侧挂并汇回。
4. K3 KDA state 呈 128×128 方块 + ×96 叠片；专家格 896 亮 16。
5. 滚轮缩放、拖拽平移、章节点击、播放流动均可用；四模型 × 两模式 × K3 双分支
   渲染无 console 报错，播放路径可走完。

## Stages

### Stage 1: 几何解析与变尺寸布局
**Goal**: `nodeGeometry()` + 变行高/spine 中轴的 `computeLayout()`
**Success Criteria**: 校准点尺寸 ±2px；四模型布局不重叠；残差在中轴
**Tests**: `node --check js/walkthrough.js`；浏览器目测
**Status**: Not Started

### Stage 2: 节点渲染
**Goal**: 矩阵网格 pattern、叠片、专家格、维度刻度；删 `matrixCells()`
**Success Criteria**: 验收 2 / 4
**Tests**: 截图对照校准点
**Status**: Not Started

### Stage 3: 全模型柱
**Goal**: 压缩层堆（真实种类交错）+ Embedding/Output 真实比例收尾
**Success Criteria**: 验收 1
**Tests**: 四模型全景截图
**Status**: Not Started

### Stage 4: 导航与播放
**Goal**: 滚轮缩放/拖拽平移/双击适应/章节锚点/播放流动边
**Success Criteria**: 验收 5 前半
**Tests**: 浏览器手测
**Status**: Not Started

### Stage 5: 图元数据与图例
**Goal**: `extra.stack/stackNote/spine` 补齐；legend 一行
**Success Criteria**: ×N 与架构数一致（96/896/64/…）
**Tests**: 逐模型核对
**Status**: Not Started

### Stage 6: 浏览器验证
**Goal**: 四模型 × Prefill/Decode × K3 双分支 + 播放录屏
**Success Criteria**: 验收 5，console 干净
**Tests**: `python3 -m http.server 8000` 手测
**Status**: Not Started
