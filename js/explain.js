(function () {
  const GROUP_SECTION = ["embed", "attn", "moe", "out"];
  const SECTION_TITLE = {
    intro: "Intro",
    embed: "Embedding",
    attn: "Attention",
    moe: "MoE / FFN",
    out: "Output",
  };

  const INTRO = [
    { id: "read", section: "intro", title: "怎么读图" },
    { id: "overview", section: "intro", title: null },
    { id: "prefill", section: "intro", title: "Prefill" },
    { id: "decode", section: "intro", title: "Decode" },
    { id: "layers", section: "intro", title: "其余层" },
  ];

  const READ = {
    kicker: "INTRO · 读图",
    title: "怎么读这张图",
    what: "每个盒子是一个张量或算子。目录是结构索引：点目录或盒子，下面四行说明当前这一块。播放按依赖走完 Prefill，再走 Decode。",
    map: "宽 ∝ log₂ 列维（隐层 / latent / 词表）。高 ∝ log₂ 行维（B×T）。叠片 ×N = heads 或 experts。网格 = 权重矩阵。中轴是 residual。中间框是代表层，其下薄片是其余层。",
    phase: "切到 Decode 时，激活盒子变矮（T→1），cache 盒子仍按序列长 S 画。颜色只标种类：绿激活、蓝权重、紫算子、橙 cache、红通信。",
    cost: "图不画时间。选中后看本页「代价」和下面的 FLOPs / 读写 / 通信。最慢的盒子在图上偏橙。",
  };

  const PREFILL = {
    kicker: "INTRO · PREFILL",
    title: "Prefill · miss tokens",
    what: "输入里 cache 未命中的一段 [B, T, H] 过完全部层，写入 KV 或 recurrent state，得到第一个输出 token。这段时间是 TTFT。",
    map: "激活盒子按 B×T 长高。Embedding 与 Output 各做一次；中间 Decoder 框重复 layers 次，图上只展开一层。",
    phase: "T = T_in × (1 − hit)。命中段不重算，但仍按输入 token 计费。Kimi K3 要分别走 KDA 层与 Gated MLA 层。",
    cost: "大 GEMM 和整段 Attention 吃 FLOPS。MoE 按本卡命中的 expert 整块读权重。跨节点 all-to-all 体积随 T 涨。",
  };

  const DECODE = {
    kicker: "INTRO · DECODE",
    title: "Decode · token 2+",
    what: "之后每个新 token 只走 [B, 1, H]，读并追加 cache。每步时间是 TPOT；集群 output tok/s ≈ B / TPOT。",
    map: "激活变矮。cache 仍按已生成长度 S 画。专家格仍然亮 Top-K，因为每步都要路由。",
    phase: "不再扫整段输入。KDA 只更新固定 state；MLA / GQA 读长度为 S 的 KV。输出头仍只看最后一个位置。",
    cost: "行少，GEMM 常变成读权重。小 batch 时 MoE 每步整块读被命中的 expert，不随 batch 线性摊薄。长上下文时 KV 带宽往往决定 TPOT。",
  };

  function layersPage(model) {
    const d = model.dims;
    const note = d.kdaLayers
      ? `${d.kdaLayers} 层 KDA + ${d.mlaLayers} 层 Gated MLA，按层号交错，不会同层并存。`
      : `${d.denseLayers || 0} 层 dense FFN + ${d.moeLayers || d.layers} 层 MoE。`;
    return {
      kicker: "INTRO · 层堆",
      title: `其余 ${Math.max(0, d.layers - 1)} 层`,
      what: `中间框是代表层。其下每条薄片是其余一层，颜色 = 层类型。${note}`,
      map: "薄片宽度 = 隐层 H。点击薄片或本条目录，视口回到代表层。层数与种类来自模型 config，不是装饰。",
      phase: "Prefill 与 Decode 都要走完全部层。图上只算一层的代价，roofline 再乘层数（once 节点不乘）。",
      cost: "总时间 ≈ 代表层 × 层数。Kimi K3 要把 KDA 层与 MLA 层分开加总。",
    };
  }

  const MODELS = {
    "glm-52": {
      overview: {
        kicker: "INTRO · GLM-5.2",
        title: "GLM-5.2 · 744B / 40B",
        what: "78 层 decoder。Attention 是 MLA，并用 DSA 只读 Top-2048 个历史位置。每 4 个稀疏层里 1 个算满 indexer，后 3 层 IndexShare 复用。FFN 是 Top-8 / 256 MoE + 1 个 shared expert。",
        map: "中轴：Hidden → Attention residual → Block residual → Logits。左侧 MLA 与 DSA indexer 并行；右侧专家格 256 亮 8。",
        phase: "Prefill 写 compressed KV（512+64）和 indexer key。Decode 只读 DSA 选出的位置，不读展开 K/V。",
        cost: "权重 FP8。一份 replica 通常 TP∩EP 重叠在一个 8 卡节点。对照 GPUStack 8×H200、8K/1K、512 并发。",
      },
      chapters: [
        { id: "embed", section: "embed", title: "Embedding", nodeIds: ["token_ids", "embedding", "hidden_in"] },
        { id: "dsa", section: "attn", title: "DSA indexer", nodeIds: ["attn_norm", "q_a", "q_resid", "index_q", "index_k", "index_scores", "index_topk", "index_share"] },
        { id: "mla", section: "attn", title: "Sparse MLA", nodeIds: ["q_up", "kv_down", "kv_cache", "kv_up", "rope", "sparse_attention", "attn_out", "attn_residual"] },
        { id: "moe", section: "moe", title: "MoE", nodeIds: ["ffn_norm", "router", "router_topk", "routed_experts", "shared_expert", "moe_combine", "block_residual"] },
        { id: "ep", section: "moe", title: "Expert parallel", nodeIds: ["dispatch", "all_to_all", "expert_compute", "gather"], epOnly: true },
        { id: "out", section: "out", title: "Output head", nodeIds: ["final_norm", "lm_head", "logits"] },
      ],
      pages: {
        embed: {
          kicker: "EMBEDDING",
          title: "词表 → Hidden",
          what: "154,880 词表查到 H=6,144。这是 residual stream 的起点，之后每层加减都回到这条宽度。",
          map: "Embedding 是 154,880 × 6,144 的矩阵。Hidden 高随 B×T，宽钉在 6,144。",
          phase: "Prefill 查 miss 段全部 token。Decode 只查新采样的 1 个 token。",
          cost: "查表一次，通常不是瓶颈。词表宽，盒子在图上偏宽。",
        },
        dsa: {
          kicker: "ATTENTION · DSA",
          title: "DSA indexer",
          what: "用轻量 Q/K 给历史位置打分，每 query 只留 Top-2048。IndexShare：4 层里 1 层算满 indexer，后 3 层复用同一套 indices。",
          map: "Indexer Q 叠 32 头；scores 的高随 T×S。IndexShare 是橙 cache，不是新算子。",
          phase: "Prefill 写 indexer key，并算 scores。Decode 对当前 token 打分，只从 cache 读 key。",
          cost: "比主 MLA 便宜，但 scores 随 S 涨。IndexShare 把 3/4 稀疏层的 indexer FLOPs 去掉。",
        },
        mla: {
          kicker: "ATTENTION · MLA",
          title: "Sparse MLA",
          what: "Q 先压到 2,048 再升到 64×256。KV 压成 512+64 写入 cache，用时再展开。Attention 只读 DSA 选出的位置。",
          map: "KV cache 高随 S、宽 576。64 头画成叠片。残差回到中轴 6,144。",
          phase: "Prefill 写 compressed KV，对 miss 段做 masked MLA。Decode 读 cache，不存展开 K/V。",
          cost: "Decode 带宽按 576 维 latent 计，不是按展开头。稀疏把 Attention 从 S 收到 2048。",
        },
        moe: {
          kicker: "MOE",
          title: "Top-8 / 256",
          what: "Router 给 256 个 expert 打分，每 token 取 8 个做 SwiGLU（6,144→2,048→6,144）。1 个 shared expert 始终执行。",
          map: "专家格最多画 128 格，256 里按比例亮 8。中轴 residual 在 combine 之后。",
          phase: "Prefill 与 Decode 都路由。Decode 每步都要重新选 expert。",
          cost: "小 batch Decode 按本卡命中的 distinct expert 整块读权重，min(B×1×8/EP, 256/EP) 个，不随 T 摊薄。",
        },
        ep: {
          kicker: "MOE · EP",
          title: "Expert parallel",
          what: "EP>1 时 token 按 expert 分组，all-to-all 换到持有该 expert 的卡，算完再 gather 回原顺序。",
          map: "红边是通信。专家格在本地卡上，只亮本卡分到的 expert。",
          phase: "Prefill 交换体积随 T×Top-K。Decode 每 token 一次，体积小但延迟在路径上。",
          cost: "同节点走 NVLink，跨节点走 RoCE。时间取 max(算、HBM、互联)。",
        },
        out: {
          kicker: "OUTPUT",
          title: "LM head",
          what: "最后一层 residual 做 RMSNorm，只取最后一个位置，乘 6,144 × 154,880 得到 logits，再采样。",
          map: "在层堆之下。LM head 是整张图最宽的权重矩阵之一。Logits 钉在中轴。",
          phase: "Prefill 与 Decode 都只看最后位置。不随层重复。",
          cost: "大 GEMM，但只做一次。Decode 时常被读词表权重卡住。",
        },
      },
      nodes: {
        index_share: {
          what: "把上一 full indexer 的 Top-2048 indices 复用到本层。不重新打分。",
          map: "橙 cache，形状与 Top-K indices 相同。",
          phase: "Prefill / Decode 都读。本层若是 full indexer 则走左边的计算路径。",
          cost: "几乎只有一次小读取。省掉 3/4 稀疏层的 indexer FLOPs。",
        },
        kv_cache: {
          what: "Compressed KV：512 latent + 64 RoPE key。这是 Decode 读历史的唯一入口。",
          map: "高 ∝ B×S，宽 576。Decode 时它比变矮的激活更高。",
          phase: "Prefill 写入 miss 段。Decode 读 S，并追加当前 token。",
          cost: "Decode 读字节 = B × S × 576 × dtype。长上下文时这是 Attention 的带宽项。",
        },
        sparse_attention: {
          what: "64 头 MLA，mask 掉 DSA 未选中的位置，只对 Top-2048 做 QKᵀ→V。",
          map: "叠 64 片。高随 T，宽是 head dim。",
          phase: "Prefill 对 miss 段每个 query 选 2048。Decode 一个 query 对历史 2048。",
          cost: "计算从 O(T×S) 收到 O(T×2048)。仍要加上 KV 展开的 GEMM。",
        },
      },
    },

    "kimi-k3": {
      overview: {
        kicker: "INTRO · KIMI K3",
        title: "Kimi K3 · 2.8T / 104B",
        what: "93 层 decoder：69 层 KDA + 24 层 Gated MLA，同层只走一种。FFN 是 LatentMoE——先压到 3,584，再 Top-16 / 896，两个 shared expert 始终开。",
        map: "中轴：Hidden → Attention output → Block output → Logits。切 KDA / Gated MLA 只换 Attention 一侧。专家格 896 亮 16。薄片按 KDA/MLA 交错着色。",
        phase: "KDA 更新固定大小 recurrent state，与序列长无关。MLA 才读写 compressed KV。界面上的层开关对应真实层类型，不是两种同时算。",
        cost: "Attention 权重 BF16；routed expert 是 MXFP4。H200 上 MXFP4 走 marlin，按 BF16 算力计。一份 replica 通常要 16 卡。",
      },
      prefill: {
        what: "miss 段 [B, T, H] 过 93 层。KDA 层写 96×128×128 state；MLA 层写 512+64 KV。出第一个 token，时间是 TTFT。",
        map: "激活按 B×T 画高。目录里 Attention 只列出当前层类型（KDA 或 Gated MLA）。",
        phase: "两种层不会在同一层同时执行。播放路径会先走完 Prefill 的 KDA，再走 Prefill 的 MLA。",
        cost: "KDA Prefill 仍对 T 做短卷积和 delta update，但不做 T×T。MLA Prefill 是完整 Attention，通常更贵。",
      },
      decode: {
        what: "当前 token [B, 1, H]。KDA 只更新固定 state；MLA 读长度为 S 的 KV。每步是 TPOT。",
        map: "激活变矮。KDA state 叠 96 片，大小不随 S 变。MLA cache 高随 S。",
        phase: "69/93 层走 KDA，所以长上下文 Decode 多数层不受 S 带宽拖累。MLA 层仍随 S 涨。",
        cost: "小 batch 时瓶颈常在 LatentMoE 读 16 个 expert 的 MXFP4 权重。MLA 层在长 S 时变成 HBM。",
      },
      layers: {
        what: "代表层按当前开关展开 KDA 或 Gated MLA。其余 92 层压成薄片：深色 KDA，橙色 MLA，按层号交错。",
        map: "薄片宽 = 7,168。点薄片回到代表层。交错从 dims.kdaLayers / mlaLayers 推导。",
        phase: "总前向 = 69 次 KDA + 24 次 MLA + 92 次 LatentMoE（首层 dense 除外）。",
        cost: "roofline 分别乘两种 Attention 层数，再加 MoE × 92。不要把两种 Attention 加在同一层上。",
      },
      chapters: [
        { id: "embed", section: "embed", title: "Embedding", nodeIds: ["text_ids", "embedding", "hidden_in"] },
        { id: "kda", section: "attn", title: "KDA", nodeIds: ["attn_norm", "kda_qkv", "short_conv", "kda_gate", "kda_state", "kda_output"], branch: "kda" },
        { id: "mla", section: "attn", title: "Gated MLA", nodeIds: ["attn_norm", "mla_q", "mla_kv", "mla_cache", "gated_mla"], branch: "mla" },
        { id: "attn-out", section: "attn", title: "Attention 输出", nodeIds: ["attention_output"] },
        { id: "moe", section: "moe", title: "LatentMoE", nodeIds: ["ffn_norm", "latent_down", "router", "top16", "routed_experts", "shared_experts", "latent_moe_combine", "latent_norm", "latent_up", "attnres_bank", "block_output"] },
        { id: "ep", section: "moe", title: "Expert parallel", nodeIds: ["dispatch", "all_to_all", "expert_compute", "inverse_gather"], epOnly: true },
        { id: "out", section: "out", title: "Output head", nodeIds: ["final_norm", "lm_head", "logits"] },
      ],
      pages: {
        embed: {
          kicker: "EMBEDDING",
          title: "词表 → Hidden",
          what: "163,840 词表查到 H=7,168。视觉塔不计入文本吞吐。这是 residual 中轴的起点。",
          map: "Embedding 163,840 × 7,168。Hidden 宽钉在 7,168，高随 B×T。",
          phase: "Prefill 查 miss 段。Decode 查 1 个新 token。",
          cost: "一次查表。词表宽，盒子偏宽，通常不是瓶颈。",
        },
        kda: {
          kicker: "ATTENTION · KDA",
          title: "Kimi Delta Attention",
          what: "96 头、每头 128 维。Q/K/V 先做 kernel=4 的因果短卷积，再用 gate/beta 做 delta-rule 更新固定 state。不生成 T×T 分数矩阵。",
          map: "QKV 与 state 叠 96 片。state 是 128×128，高不随 T 或 S 变。输出投影回 7,168 中轴。",
          phase: "Prefill 对整段 T 逐步更新 state。Decode 只做一步：读 state → 更新 → 写回。",
          cost: "时间与 S 无关。Prefill 仍随 T 涨（短卷积 + 逐步 delta）。Decode 几乎是常数，带宽读那份 96×128×128 state。",
        },
        mla: {
          kicker: "ATTENTION · MLA",
          title: "Gated MLA",
          what: "Q 走 1,536 latent；KV 压成 512+64。完整 MLA 后再乘 output gate。只有这 24 层读长序列 KV。",
          map: "cache 高随 S、宽 576。Gated MLA 叠 96 头。切到 KDA 时这些盒子从 DAG 消失。",
          phase: "Prefill 写 compressed KV 并做满 Attention。Decode 读 S，追加 1。",
          cost: "24/93 层。长上下文 Decode 的 Attention 带宽几乎都在这里。KDA 层分担不了这笔。",
        },
        "attn-out": {
          kicker: "ATTENTION",
          title: "Attention 输出",
          what: "当前层类型的输出。KDA 与 MLA 在图上是两条互斥边，汇到同一中轴盒子。",
          map: "中轴、宽 7,168。上游只亮当前分支。",
          phase: "形状在 Prefill / Decode 都是 [B, T, 7,168]，Decode 的 T=1。",
          cost: "它本身是 residual 汇合，代价在上游分支。",
        },
        moe: {
          kicker: "MOE · LATENT",
          title: "LatentMoE Top-16 / 896",
          what: "先把 7,168 压到 3,584，再路由 896 个 latent expert，每 token 16 个，SiTU 3,584→3,072→3,584。2 个 shared expert 始终算。再 RMSNorm、升回 7,168。AttnRes 每 12 层存一次 prefix residual。",
          map: "down/up 是窄矩阵。专家格 896 亮 16。Block output 回到中轴 7,168。",
          phase: "每层都走。Decode 每 token 重新路由。AttnRes 读的是跨层残差，不是 KV。",
          cost: "Decode 小 batch：读 16 个 MXFP4 expert 整块权重。这是 K3 小并发 Decode 贵的主因。Prefill 可被 T 摊薄。",
        },
        ep: {
          kicker: "MOE · EP",
          title: "Expert parallel",
          what: "EP>1 时按 expert 分组 3,584 维 latent rows，all-to-all 到持有该 expert 的卡，算完 inverse gather。",
          map: "红通信边。本地专家格只亮本卡分片。",
          phase: "Prefill 体积随 T×16。Decode 每步一次。",
          cost: "K3 在 H200 上 replica 常跨两个 8 卡节点，all-to-all 会掉到 RoCE。",
        },
        out: {
          kicker: "OUTPUT",
          title: "LM head",
          what: "最后位置 RMSNorm，乘 7,168 × 163,840，采样一个 token。",
          map: "层堆之下。LM head 是最宽的矩阵之一。",
          phase: "只做一次，只看最后位置。",
          cost: "大 GEMM 一次。Decode 时常是读词表权重。",
        },
      },
      nodes: {
        kda_state: {
          what: "Delta-rule 隐状态：每头一张 128×128 表。用它代替 KV cache。",
          map: "叠 96 片。片的宽高是 128，与 T、S 无关，所以 Decode 时它比激活更高。",
          phase: "Prefill 从左到右扫 T，逐步写。Decode 读-改-写一步。",
          cost: "字节 = B × 96 × 128 × 128 × dtype。不随上下文变。这是 KDA 层 Decode 近常数的原因。",
        },
        short_conv: {
          what: "Q、K、V 三条流各做 kernel=4 的因果 conv1d，再进 delta update。",
          map: "算子盒，不画成矩阵。",
          phase: "Prefill 对整段做因果卷积。Decode 用长 4 的环形缓冲。",
          cost: "相对 GEMM 小。Prefill 随 T 线性。",
        },
        mla_cache: {
          what: "仅 MLA 层的 compressed KV（512+64）。KDA 层不读它。",
          map: "高 ∝ B×S，宽 576。",
          phase: "Prefill 写。Decode 读 S、追加 1。",
          cost: "24 层 × B × S × 576 × dtype。长上下文时这是 K3 Attention 的 HBM 项。",
        },
        latent_down: {
          what: "7,168 → 3,584。路由和 expert 都在这半宽上算，所以叫 LatentMoE。",
          map: "激活变窄。后续专家格的列维是 3,072，不是 7,168。",
          phase: "每层、每个 token 都做。",
          cost: "一次 GEMM。Prefill 吃算力；Decode 吃读权重。",
        },
        routed_experts: {
          what: "896 个 latent expert 里激活 16 个。每个是 3,584→3,072→3,584 的 SiTU。",
          map: "格点 = expert。亮的是 Top-16。超过 128 个时按比例抽样画格。",
          phase: "每步重选。Decode 不能复用上一 token 的 expert 集合。",
          cost: "读 min(B×T×16/EP, 896/EP) 个 expert 的整块 MXFP4 权重。小 B、T=1 时几乎读满 16 份。",
        },
        attnres_bank: {
          what: "每 12 层存一份 attention residual 的前缀和，供后续 block 使用。",
          map: "橙 cache，宽 7,168。不是 KV。",
          phase: "Prefill 写块。Decode 读最近的 bank。",
          cost: "相对 MoE 与 MLA cache 很小。",
        },
      },
    },

    "kimi-k25": {
      overview: {
        kicker: "INTRO · KIMI K2.5",
        title: "Kimi K2.5 · 1T / 32B",
        what: "61 层。Attention 是标准 MLA：Q 走 1,536 latent，KV 走 512+64。FFN 是 Top-8 / 384 MoE + 1 shared。Routed expert 是 INT4，Attention / shared / 首层 dense / LM head 仍是 BF16。",
        map: "中轴 Hidden → Attention residual → Block residual。MLA 在左，专家格 384 亮 8。",
        phase: "Prefill 写 compressed KV。Decode 按 serving 的 absorbed 512+64 计 HBM，不按 Hugging Face 展开的 K/V。",
        cost: "INT4 只打 routed GEMM。同一张卡上，这些 expert 比 BF16 Attention 更吃带宽折算。",
      },
      chapters: [
        { id: "embed", section: "embed", title: "Embedding", nodeIds: ["text_ids", "embedding", "hidden_in"] },
        { id: "mla", section: "attn", title: "MLA", nodeIds: ["attn_norm", "q_down", "q_up", "kv_down", "kv_cache", "kv_up", "mla_attn", "o_proj", "attn_residual"] },
        { id: "moe", section: "moe", title: "MoE", nodeIds: ["ffn_norm", "router", "topk", "routed_experts", "shared_expert", "moe_combine", "block_residual"] },
        { id: "ep", section: "moe", title: "Expert parallel", nodeIds: ["dispatch", "all_to_all", "expert_compute", "gather"], epOnly: true },
        { id: "out", section: "out", title: "Output head", nodeIds: ["final_norm", "lm_head", "logits"] },
      ],
      pages: {
        embed: {
          kicker: "EMBEDDING",
          title: "词表 → Hidden",
          what: "163,840 → H=7,168。residual 中轴从此开始。",
          map: "宽矩阵 × 窄 Hidden。Hidden 高随 B×T。",
          phase: "Prefill 整段查表。Decode 查 1。",
          cost: "一次 lookup。",
        },
        mla: {
          kicker: "ATTENTION · MLA",
          title: "MLA",
          what: "Q：7,168→1,536→64×192（128 NoPE + 64 RoPE）。KV：压成 512+64，用时升到 64×256。QKᵀ 在 NoPE 与 RoPE 子空间组合。",
          map: "cache 高随 S、宽 576。64 头叠片。残差回 7,168。",
          phase: "Prefill 写 latent 并做满 Attention。Decode 读 absorbed cache；KV up 可吸进 kernel，不单独占一步 HBM。",
          cost: "Decode 带宽按 576 维计。这是长上下文 TPOT 的主项之一。",
        },
        moe: {
          kicker: "MOE",
          title: "Top-8 / 384",
          what: "384 个 SwiGLU expert，每 token 8 个，宽 7,168→2,048→7,168。1 个 shared 始终算。Router 的 correction bias 只改选择，不改分数本身。",
          map: "专家格 384 亮 8。",
          phase: "每步路由。",
          cost: "INT4 routed 权重。小 batch Decode 读 8 个 expert 整块。",
        },
        ep: {
          kicker: "MOE · EP",
          title: "Expert parallel",
          what: "发布 config 的 ep_size=1。分布式部署时可覆盖。EP>1 才走 dispatch / all-to-all / gather。",
          map: "红通信边只在 EP>1 出现。",
          phase: "Prefill 体积随 T。Decode 每 token 一次。",
          cost: "同节点 NVLink，跨节点 RoCE。",
        },
        out: {
          kicker: "OUTPUT",
          title: "LM head",
          what: "最后位置 → 163,840 logits。BF16，不是 INT4。",
          map: "层堆下的宽矩阵。",
          phase: "一次，最后位置。",
          cost: "读 7,168×163,840。Decode 可能和 MoE 抢 HBM。",
        },
      },
      nodes: {
        kv_cache: {
          what: "Serving 按 absorbed 512+64 计。Hugging Face reference 会缓存展开 K/V，本图不按那条计带宽。",
          map: "高 ∝ B×S，宽 576。",
          phase: "Prefill 写。Decode 读 S。",
          cost: "B × S × 576 × FP8。不要用展开后的 64×256 去估 Decode HBM。",
        },
      },
    },

    "minimax-m3": {
      overview: {
        kicker: "INTRO · MINIMAX M3",
        title: "MiniMax M3 · 428B / 23B",
        what: "60 层。代表层是 block-sparse GQA：64 个 Q 头、4 个 KV 头。Lightning indexer 选 16 个 128-token 块，再加 1 个 local 块。前 3 层 dense FFN，其余 Top-4 / 128 MoE。权重 BF16。",
        map: "中轴 Hidden → Attention residual → Block residual。Indexer 叠 4 头。专家格 128 亮 4。薄片前 3 条是 dense。",
        phase: "KV 按全量 paged 占显存。Attention 计算只读被选中的块。",
        cost: "稀疏省的是 FLOPs，不是 KV 显存。Decode 仍按全 S 占 HBM；算力按 16×128+local。",
      },
      chapters: [
        { id: "embed", section: "embed", title: "Embedding", nodeIds: ["text_ids", "embedding", "hidden_in"] },
        { id: "index", section: "attn", title: "Lightning indexer", nodeIds: ["attn_norm", "indexer", "index_topk"] },
        { id: "gqa", section: "attn", title: "Sparse GQA", nodeIds: ["q_proj", "kv_proj", "kv_cache", "sparse_gqa", "attn_out", "attn_residual"] },
        { id: "moe", section: "moe", title: "MoE", nodeIds: ["ffn_norm", "router", "top4", "routed_experts", "shared_expert", "moe_combine", "block_residual"] },
        { id: "ep", section: "moe", title: "Expert parallel", nodeIds: ["dispatch", "all_to_all", "expert_compute", "inverse_gather"], epOnly: true },
        { id: "out", section: "out", title: "Output head", nodeIds: ["final_norm", "lm_head", "logits"] },
      ],
      pages: {
        embed: {
          kicker: "EMBEDDING",
          title: "词表 → Hidden",
          what: "200,064 → H=6,144。",
          map: "最宽的词表矩阵之一。Hidden 钉在 6,144。",
          phase: "Prefill 整段。Decode 1 token。",
          cost: "一次查表。",
        },
        index: {
          kicker: "ATTENTION · INDEX",
          title: "Lightning indexer",
          what: "4 头、每头 128 维，给 KV 块打分。选出 Top-16 个 128-token 块，再加上当前 local 块。稀疏单位是块，不是单个 token。",
          map: "Indexer 叠 4 片。Top-16 是 indices，不是大激活。",
          phase: "Prefill 对 miss 段每个 query 选块。Decode 对当前 token 选块。",
          cost: "比主 GQA 便宜。选出的块数固定，所以主 Attention 计算不随 S 线性涨。",
        },
        gqa: {
          kicker: "ATTENTION · GQA",
          title: "Sparse GQA",
          what: "64 个 Q 头共享 4 组 KV。计算只读 indexer 选出的 paged 块。显存仍按全序列 4×256 计。",
          map: "KV cache 高随 S。Q 叠 64。残差回 6,144。",
          phase: "Prefill 写全量 paged KV，计算只扫选中块。Decode 同样：显存全 S，计算选中块。",
          cost: "HBM 占用量按 S；FLOPs 按 16×128+local。长上下文先撞显存，再撞算力。",
        },
        moe: {
          kicker: "MOE",
          title: "Top-4 / 128",
          what: "128 个 SwiGLU-OAI，每 token 4 个，6,144→3,072→6,144。1 个 shared。前 3 层不用 MoE，走 dense（薄片里的蓝层）。",
          map: "专家格 128 亮 4。",
          phase: "MoE 层每步路由。Dense 层没有这组盒子。",
          cost: "BF16。小 batch Decode 读 4 个 expert 整块。激活比 GLM 少，所以 MoE 项通常轻于稀疏 GQA 的 KV。",
        },
        ep: {
          kicker: "MOE · EP",
          title: "Expert parallel",
          what: "EP>1 时 Top-4 rows 做 all-to-all。",
          map: "红边。本地 128/EP 个 expert。",
          phase: "Prefill 随 T。Decode 每 token。",
          cost: "同节点 NVLink，跨节点 RoCE。",
        },
        out: {
          kicker: "OUTPUT",
          title: "LM head",
          what: "最后位置 Gemma RMSNorm，乘 6,144 × 200,064。",
          map: "层堆下的宽矩阵。",
          phase: "一次。",
          cost: "词表比其他三个模型都宽。Decode 读权重明显。",
        },
      },
      nodes: {
        kv_cache: {
          what: "Paged GQA KV，按全量 S 占显存。计算不读未选中的页。",
          map: "高 ∝ B×S，叠 4 个 KV 头。",
          phase: "Prefill 写入全部 miss token。Decode 追加 1，读选中块。",
          cost: "显存 = B × S × 4 × 256 × BF16。这是 batch 上限，不是 FLOPs。",
        },
        sparse_gqa: {
          what: "64 头 GQA，只 attend indexer 选出的块。",
          map: "叠 64。宽 128。",
          phase: "Prefill / Decode 都只读选中块。",
          cost: "FLOPs ∝ T × (16×128 + local) × heads。与全量 S 解耦。",
        },
      },
    },
  };

  function groupOf(node) {
    return window.BallparkWalkthrough.groupOf(node);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function spec(model) {
    return MODELS[model.id] || MODELS["glm-52"];
  }

  function visibleChapters(model, nodes, { branch, ep }) {
    const ids = new Set(nodes.map((node) => node.id));
    return spec(model).chapters.filter((chapter) => {
      if (chapter.branch && branch && chapter.branch !== branch) return false;
      if (chapter.epOnly && !(ep > 1)) return false;
      return (chapter.nodeIds || []).some((id) => ids.has(id));
    });
  }

  function buildToc(model, nodes, opts) {
    const intro = INTRO.map((item) => ({
      ...item,
      title: item.id === "overview" ? model.name : item.title,
    }));
    const chapters = visibleChapters(model, nodes, opts).map((chapter) => ({
      id: chapter.id,
      section: chapter.section,
      title: chapter.title,
      nodeIds: chapter.nodeIds,
    }));
    return [...intro, ...chapters];
  }

  function chapterOfNode(model, nodeId, nodes, opts) {
    return visibleChapters(model, nodes, opts).find((chapter) => (chapter.nodeIds || []).includes(nodeId)) || null;
  }

  function defaultPhase(node, mode) {
    if (node.once) return "只在 Embedding 或 Output 做一次，不随层重复。";
    if (node.cacheRole === "write") {
      return mode === "prefill" ? "Prefill 把 miss 段写入 cache。" : "Decode 追加当前 token。";
    }
    if (node.cacheRole === "readAppend") {
      return mode === "prefill" ? "Prefill 建立这份 cache。" : "Decode 读历史并追加。";
    }
    return mode === "prefill"
      ? "对 miss 段整段做，形状里的 T 是未命中长度。"
      : "只对当前 1 个 token 做。T=1。";
  }

  function defaultCost(node) {
    const id = node.costId || "none";
    if (id === "moe_routed") return "按本卡命中的 distinct expert 整块读权重。小 batch Decode 不随 T 摊薄。";
    if (id === "moe_shared") return "每个 token 都算。相对 routed 小，但是必付。";
    if (id === "all_to_all") return "时间取互联。Prefill 随 T；Decode 每 token 一次。";
    if (id.startsWith("kv_cache") || id === "attn_kda") return "Decode 常被这份 cache 的带宽卡住。";
    if (id.startsWith("attn") || id === "indexer_scan" || id === "m3_indexer") {
      return "Prefill 偏算力。Decode 偏读 cache。";
    }
    if (id === "gemm" || id === "embed" || id === "kda_qkv" || id === "kda_gate" || id === "mla_q" || id === "indexer_q" || id === "indexer_k") {
      return "Prefill 吃 FLOPS。Decode 行少，常变成读权重。";
    }
    if (id === "rms" || id === "rope" || id === "kda_conv") return "相对主 GEMM / Attention 通常可忽略。";
    if (id === "none") return "路由或重排，几乎不进 roofline。";
    return "时间取 max(算力, HBM, 互联)。";
  }

  function defaultMap(node, ctx, mode) {
    const shape = window.BallparkWalkthrough.resolveShape(node.shape, ctx);
    if (node.dIn && node.dOut) {
      return `权重 ${node.dIn.toLocaleString()} × ${node.dOut.toLocaleString()}。网格 = 矩阵。盒宽 ∝ 列维。`;
    }
    if (node.stack && node.stackActive) {
      return `形状 ${shape}。格点 = expert，亮 ${node.stackActive} / ${node.stack}。`;
    }
    if (node.stack > 1) {
      return `形状 ${shape}。叠 ${node.stack} 片。高随 ${mode === "prefill" ? "B×T" : "B×1"}。`;
    }
    return `形状 ${shape}。盒高随 ${mode === "prefill" ? "B×T" : "B×1"}，宽随列维。`;
  }

  function mergePage(base, over) {
    if (!over) return base;
    return {
      kicker: over.kicker || base.kicker,
      title: over.title || base.title,
      what: over.what || base.what,
      map: over.map || base.map,
      phase: over.phase || base.phase,
      cost: over.cost || base.cost,
    };
  }

  function introPage(id, model, mode) {
    if (id === "read") return READ;
    if (id === "prefill") return mergePage(PREFILL, spec(model).prefill);
    if (id === "decode") return mergePage(DECODE, spec(model).decode);
    if (id === "layers") return mergePage(layersPage(model), spec(model).layers);
    return spec(model).overview;
  }

  function page(model, opts) {
    const { explainId, node, nodes, ctx, mode, branch, ep } = opts;
    const id = explainId || (node ? `node:${node.id}` : "overview");

    if (!id.startsWith("node:")) {
      const chapter = spec(model).pages?.[id];
      if (chapter) return chapter;
      return introPage(id, model, mode);
    }

    const current = node || nodes.find((item) => item.id === id.slice(5));
    if (!current) return introPage("overview", model, mode);

    const chapter = chapterOfNode(model, current.id, nodes, { branch, ep });
    const section = GROUP_SECTION[groupOf(current)] || "attn";
    const base = {
      kicker: `${(chapter && spec(model).pages?.[chapter.id]?.kicker) || SECTION_TITLE[section]} · ${current.kind}`,
      title: current.label,
      what: current.detail,
      map: defaultMap(current, ctx, mode),
      phase: defaultPhase(current, mode),
      cost: defaultCost(current),
    };
    return mergePage(base, spec(model).nodes?.[current.id]);
  }

  function chapterIdFor(model, explainId, nodeId, nodes, opts) {
    if (explainId && !explainId.startsWith("node:")) return explainId;
    const id = nodeId || (explainId && explainId.startsWith("node:") ? explainId.slice(5) : null);
    if (!id) return "overview";
    return chapterOfNode(model, id, nodes, opts)?.id || "overview";
  }

  window.BallparkExplain = {
    SECTION_TITLE,
    GROUP_SECTION,
    buildToc,
    page,
    chapterOfNode,
    chapterIdFor,
    escapeHtml,
  };
}());
