# Inference Ballpark

用模型单元架构和硬件 roofline 估算 Prefill / Decode 吞吐。无构建依赖。

```bash
python3 -m http.server 8000
```

然后打开 <http://localhost:8000>。

## 部署（Cloudflare Pages）

无构建，纯静态。把 `index.html` 和 `js/` 拷进一个干净目录再上传（避免把仓库文档发上去）。`wrangler.toml` 已配好项目名与 `pages_build_output_dir = .deploy`：

```bash
rm -rf .deploy && mkdir .deploy && cp index.html .deploy/ && cp -r js .deploy/js
npx wrangler@latest pages deploy --branch=main
```

需要环境变量 `CLOUDFLARE_API_TOKEN` 和 `CLOUDFLARE_ACCOUNT_ID`。首次建项目：`npx wrangler pages project create inference-ballpark --production-branch=main`。自定义域名在 Cloudflare 面板（Pages → Custom domains）绑定，或建一条代理 CNAME 指向 `inference-ballpark.pages.dev`。线上：<https://inference-viz.latentvibe.com>。

## 做什么

页面从左到右是架构和配置；**吞吐结果在底部抽屉**，默认收起只露出关键数字，点底栏或按 Esc 展开/收起。改请求或硬件，底栏 TTFT / TPOT / tok/s / TPM 立刻变。

左侧是自上而下的架构图：Embedding → 带框的 Decoder Layer（Attention + MoE）→ Output Head。长 residual 绕到侧边。右侧 jigs 控制 Prefill/Decode、硬件与请求。点「播放路径」会按依赖顺序走完 Prefill，再走 Decode（Kimi K3 会再切 KDA / Gated MLA）。

KV cache hit 只减少 Prefill 的 `T_miss = T_in × (1 − h)`。GPU 规格四个模型共用。TP/EP/PP 是部署参数，由模型架构和单卡显存自动推导，对照表每一行用该模型自己的摆法。节点数只表示机群规模；不足一份 replica 时会自动补齐。

- **Prefill**：整段 miss tokens `[B, T, H]` 过完全部层，写入 KV / recurrent state，得到第一个 output token。决定 TTFT。
- **Decode**：只走当前 token `[B, 1, H]`，读 cache 并追加。决定 TPOT 和通常情况下的集群 output tok/s。

| 模型 | 代表路径 |
|---|---|
| GLM-5.2 | MLA + DSA Top-2048 + IndexShare；Top-8/256 MoE |
| Kimi K3 | 69 层 KDA / 24 层 Gated MLA（界面可切换层类型）；LatentMoE Top-16/896 |
| Kimi K2.5 | MLA absorbed KV（512+64）；Top-8/384 MoE |
| MiniMax M3 | Block-sparse GQA（16×128 + local）；Top-4/128 MoE；前 3 层 dense |

## 怎么算

代价按**该模型的架构轴**切到单卡，不再把全部 FLOPs / 字节除以 `TP×EP×PP`。

```text
Dense / MLA 权重           ÷ TP（重叠布局也切）
Attention 计算 / KV        重叠 DP-attn：按序列切；否则 ÷ TP
Routed MoE 计算            ÷ EP（重叠）或 ÷ TP×EP（网格）
Shared / dense FFN         ÷ TP
MLA compressed KV          DP-attn 时按序列切，不随 EP 再切
GQA KV                     网格 ÷ min(TP, kv_heads)；DP-attn 按序列切
KDA state                  网格 heads ÷ TP；DP-attn 按序列切
EP all-to-all              同一节点走 NVLink；跨节点走 RoCE
PP                         只切权重显存和 replica 个数
```

单卡时间

```text
T = max(FLOPs_gpu / (FLOPS × MFU),
        Bytes_gpu / (HBM × η_hbm),
        Comm_NVLink / (NVLink × η_net),
        Comm_RoCE / (RoCE × η_net))
```

MFU 只打折算力（prefill 大 GEMM），η_hbm 只打折带宽（decode 多为 HBM-bound），两者分开填。

FLOPS 按 dtype 折算：只有 FP8 权重的 GEMM 吃满 FP8 峰值；BF16 减半；INT4 / MXFP4 在 Hopper 上走 marlin 解量化后按 BF16 MAC 跑，同样减半。所以同一张卡上 GLM-5.2（FP8）的 prefill 算力是 M3（BF16）的两倍。

MoE 的 expert 权重读按本卡实际命中的 distinct expert 数整块计入：`min(B×T×topk/EP, experts/EP)` 个。小 batch decode 每步都要把被路由到的 expert 权重完整读一遍，吞吐不会随 batch 线性摊薄，这正是 MoE 小并发 decode 贵的原因。

拓扑按模型推导：MLA 模型一份 replica 通常 1 个节点（TP∩EP∩DP-attn）；K3 以 KDA 为主且单节点装不下权重，replica 占 2 个节点（TP16∩EP16，Attention 走 TP）。节点内 NVLink，跨节点 RoCE。

`replica`：重叠 `max(TP, EP) × PP`，网格 `TP × EP × PP`。

全层求和得到 TTFT 与 TPOT。机群吞吐取决于能装几份 replica；装不下则只报延迟、不报 TPM。聚合 serving 用 `B / (TTFT + L_out × TPOT)`；P/D 分离时两边都要能装下 replica，吞吐取 Prefill 请求产能与 Decode token 产能的最小值。KV 显存按该模型的 `cacheKind` 下调有效 batch。

结果是 roofline 规划值。**集群 TPM 按 billable 口径**：输入 token（cache hit 段照样计费）+ 输出 token；output-only 与 req/s 在卡片小字里。tok/s/GPU 仍按 output 报，方便对照厂商 lab。对照锚点：GPUStack 8×H200 GLM-5.2 · 8K/1K · 512 并发 · output 1945 tok/s = Total TPS 17.5K；本工具同工况 B=64 给 output 257/GPU、Total TPS 18.5K，差约 6%。

## 文件

- `index.html` — 页面
- `js/models.js` — 模型维度与并行默认值
- `js/model-graphs.js` — 单元 DAG（含 cache 角色与代价 id）
- `js/roofline.js` — 逐步代价与吞吐
- `js/walkthrough.js` — 纵向 DAG 与播放高亮
- `js/app.js` — 控件与渲染
