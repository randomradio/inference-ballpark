# Inference Ballpark

用模型单元架构和硬件 roofline 估算 Prefill / Decode 吞吐。无构建依赖。

```bash
python3 -m http.server 8000
```

然后打开 <http://localhost:8000>。

## 做什么

左侧是自上而下的架构图：Embedding → 带框的 Decoder Layer（Attention + MoE）→ Output Head。长 residual 绕到侧边。右侧 jigs 控制 Prefill/Decode、硬件与请求。点「播放完整推理路径」会按依赖顺序走完 Prefill，再走 Decode（Kimi K3 会再切 KDA / Gated MLA）。

KV cache hit 只减少 Prefill 的 `T_miss = T_in × (1 − h)`。四个模型共用你填写的机群，切模型不改写 TP/EP。

- **Prefill**：整段 miss tokens `[B, T, H]` 过完全部层，写入 KV / recurrent state，得到第一个 output token。决定 TTFT。
- **Decode**：只走当前 token `[B, 1, H]`，读 cache 并追加。决定 TPOT 和通常情况下的集群 output tok/s。

| 模型 | 代表路径 |
|---|---|
| GLM-5.2 | MLA + DSA Top-2048 + IndexShare；Top-8/256 MoE |
| Kimi K3 | 69 层 KDA / 24 层 Gated MLA（界面可切换层类型）；LatentMoE Top-16/896 |
| Kimi K2.5 | MLA absorbed KV（512+64）；Top-8/384 MoE |
| MiniMax M3 | Block-sparse GQA（16×128 + local）；Top-4/128 MoE；前 3 层 dense |

## 怎么算

每一步时间取

```text
T = max(FLOPs / (FLOPS × η / replica),
        Bytes / (HBM × η / replica),
        Comm / (Net × η_net))
```

`replica = TP × EP × PP`。全层求和得到 TTFT 与 TPOT。聚合 serving 用 `B / (TTFT + L_out × TPOT)`；P/D 分离时按 P:D 切开 GPU，吞吐取 Prefill 请求产能与 Decode token 产能的最小值。KV 显存会下调有效 batch。EP>1 时图上才显示 expert all-to-all。

结果是 roofline 规划值。NVIDIA Dynamo GLM-5.2 的 68.86 tok/s/GPU 与 Kimi 社区下界只作对照，不替换公式。

## 文件

- `index.html` — 页面
- `js/models.js` — 模型维度与并行默认值
- `js/model-graphs.js` — 单元 DAG（含 cache 角色与代价 id）
- `js/roofline.js` — 逐步代价与吞吐
- `js/walkthrough.js` — 纵向 DAG 与播放高亮
- `js/app.js` — 控件与渲染
