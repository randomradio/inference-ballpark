(function () {
  const node = (id, stageIndex, label, kind, op, shape, detail, dependsOn = [], extra = {}) => ({
    id,
    stageIndex,
    label,
    kind,
    op,
    shape,
    detail,
    dependsOn,
    cacheRole: extra.cacheRole || "none",
    costId: extra.costId || "none",
    layerBranch: extra.layerBranch || null,
    epOnly: !!extra.epOnly,
    skipIfEp: !!extra.skipIfEp,
    once: !!extra.once,
    dIn: extra.dIn,
    dOut: extra.dOut,
  });

  const glm52 = {
    sourceLabel: "GLM-5.2 official config + Transformers glm_moe_dsa",
    evidenceLevel: "REFERENCE FORWARD",
    sources: [
      { label: "official config", url: "https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json" },
      { label: "reference forward", url: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py" },
    ],
    caption: "MLA 与 DSA indexer 并行。Prefill 写 compressed KV；Decode 只读 Top-2048。IndexShare 让 3/4 稀疏层复用上一 full indexer。",
    nodes: [
      node("token_ids", 0, "Token IDs", "activation", "Tokenizer output", "[{B}, {T}]", "离散 token 索引。", [], { once: true }),
      node("embedding", 0, "Embedding W", "weight", "Lookup", "[154,880 × 6,144]", "把 token ID 映射到 H=6,144。", ["token_ids"], { costId: "embed", once: true, dIn: 154880, dOut: 6144 }),
      node("hidden_in", 0, "Hidden states", "activation", "Embedded sequence", "[{B}, {T}, 6,144]", "decoder layer 输入。", ["embedding"]),

      node("attn_norm", 1, "RMSNorm", "operation", "Pre-attention norm", "[{B}, {T}, 6,144]", "self-attention 前 RMSNorm。", ["hidden_in"], { costId: "rms", dIn: 6144 }),
      node("q_a", 1, "Q down", "weight", "Linear 6,144 → 2,048", "[6,144 × 2,048]", "q_a_proj；输出也供 DSA indexer 使用。", ["attn_norm"], { costId: "gemm", dIn: 6144, dOut: 2048 }),
      node("q_resid", 1, "Q latent", "activation", "RMSNorm", "[{B}, {T}, 2,048]", "q_a_layernorm。", ["q_a"], { costId: "rms", dIn: 2048 }),
      node("index_q", 1, "Indexer Q", "weight", "wq_b", "[{B}, {T}, 32, 128]", "独立 indexer query。", ["q_resid"], { costId: "indexer_q", dIn: 2048, dOut: 4096 }),
      node("index_k", 1, "Indexer K", "weight", "wk + LN", "[{B}, {S}, 128]", "轻量 index key，Prefill 写入 indexer cache。", ["attn_norm"], { cacheRole: "write", costId: "indexer_k", dIn: 6144, dOut: 128 }),
      node("index_scores", 1, "Indexer scores", "operation", "Weighted QKᵀ", "[{B}, 32, {T}, {S}]", "对历史 token 打分。IndexShare 层可跳过。", ["index_q", "index_k"], { costId: "indexer_scan" }),
      node("index_topk", 1, "DSA Top-2,048", "operation", "topk mask", "indices [{B}, {T}, 2,048]", "被选位置 mask=0，其余 -∞。", ["index_scores"], { costId: "none" }),
      node("index_share", 1, "IndexShare", "cache", "Reuse previous Top-K", "indices [{B}, {T}, 2,048]", "每 4 个稀疏层中 1 个 full indexer，其后 3 层复用。", ["index_topk"], { cacheRole: "readAppend" }),
      node("q_up", 1, "Q up", "weight", "Linear 2,048 → 64×256", "[2,048 × 16,384]", "64 heads × (192 NoPE + 64 RoPE)。", ["q_resid"], { costId: "gemm", dIn: 2048, dOut: 16384 }),
      node("kv_down", 1, "KV down", "weight", "Joint KV compress", "[6,144 × 576]", "512 compressed KV + 64 RoPE key。Prefill 写入 cache。", ["attn_norm"], { cacheRole: "write", costId: "gemm", dIn: 6144, dOut: 576 }),
      node("kv_cache", 1, "Compressed KV cache", "cache", "512 + 64 latent", "[{B}, {S}, 576]", "Decode 从这里读历史，不读展开 K/V。", ["kv_down"], { cacheRole: "readAppend", costId: "kv_cache_mla" }),
      node("kv_up", 1, "KV up", "weight", "Linear 512 → 64×448", "[512 × 28,672]", "展开 NoPE key 与 value。", ["kv_cache"], { costId: "gemm", dIn: 512, dOut: 28672 }),
      node("rope", 1, "Interleaved RoPE", "operation", "Rotate Qᴿ / Kᴿ", "64 dims / head", "DSA reference 使用 interleaved RoPE。", ["q_up", "kv_down"], { costId: "rope" }),
      node("sparse_attention", 1, "Sparse MLA", "operation", "masked QKᵀ → V", "[{B}, 64, {T}, 256]", "只读 DSA 选出的 Top-2048。", ["q_up", "kv_up", "rope", "index_share"], { costId: "attn_sparse_mla" }),
      node("attn_out", 1, "Output Wᴼ", "weight", "Linear 16,384 → 6,144", "[16,384 × 6,144]", "投影回模型宽度。", ["sparse_attention"], { costId: "gemm", dIn: 16384, dOut: 6144 }),
      node("attn_residual", 1, "Attention residual", "activation", "Residual add", "[{B}, {T}, 6,144]", "attention + layer input。", ["attn_out", "hidden_in"]),

      node("ffn_norm", 2, "RMSNorm", "operation", "Pre-FFN norm", "[{B}, {T}, 6,144]", "MoE 前归一化。", ["attn_residual"], { costId: "rms", dIn: 6144 }),
      node("router", 2, "Router", "weight", "FP32 linear → sigmoid", "[{B}×{T}, 256]", "noaux_tc 产生 expert scores。", ["ffn_norm"], { costId: "gemm", dIn: 6144, dOut: 256 }),
      node("router_topk", 2, "Top-8 route", "operation", "select + weights", "ids [{B}×{T}, 8]", "每个 token 8 个 routed experts。", ["router"]),
      node("routed_experts", 2, "256 routed experts", "module", "Top-8 SwiGLU", "6,144 → 2,048 → 6,144", "只执行被选中的 8 个。EP=1 时在此计算。", ["router_topk", "ffn_norm"], { costId: "moe_routed", skipIfEp: true, dIn: 6144, dOut: 2048 }),
      node("shared_expert", 2, "Shared expert", "module", "Always-on SwiGLU", "6,144 → 2,048 → 6,144", "不受 Top-K 影响。", ["ffn_norm"], { costId: "moe_shared", dIn: 6144, dOut: 2048 }),
      node("moe_combine", 2, "MoE combine", "operation", "weighted + shared", "[{B}, {T}, 6,144]", "单卡 reference 路径。EP>1 时由 gather 替代。", ["routed_experts", "shared_expert", "router_topk"], { skipIfEp: true }),

      node("dispatch", 3, "Expert dispatch", "communication", "Group by expert", "[{B}×{T}×8, 6,144]", "按 expert ID 重排 token rows。", ["router_topk", "ffn_norm"], { epOnly: true, costId: "none" }),
      node("all_to_all", 3, "All-to-all", "communication", "Expert-parallel exchange", "active token rows", "仅 EP>1。Prefill 体积随 T 涨，Decode 体积小但每 token 一次。", ["dispatch"], { epOnly: true, costId: "all_to_all" }),
      node("expert_compute", 3, "Local expert compute", "module", "SwiGLU FFNs", "Top-8 routes", "每个 rank 执行本地专家。", ["all_to_all"], { epOnly: true, costId: "moe_routed", dIn: 6144, dOut: 2048 }),
      node("gather", 3, "Inverse gather", "communication", "Restore token order", "[{B}, {T}, 8, 6,144]", "专家输出返回原顺序。", ["expert_compute"], { epOnly: true, costId: "none" }),
      node("block_residual", 3, "Block residual", "activation", "FFN + residual", "[{B}, {T}, 6,144]", "进入下一层。", ["gather", "moe_combine", "shared_expert", "attn_residual"]),

      node("final_norm", 4, "Final RMSNorm", "operation", "Normalize last token", "[{B}, 6,144]", "只用最后一个位置。", ["block_residual"], { costId: "rms", once: true, dIn: 6144 }),
      node("lm_head", 4, "LM head", "weight", "Linear 6,144 → 154,880", "[6,144 × 154,880]", "投影到词表。", ["final_norm"], { costId: "gemm", once: true, dIn: 6144, dOut: 154880 }),
      node("logits", 4, "Logits", "activation", "Last-token scores", "[{B}, 154,880]", "采样一个 token。", ["lm_head"], { once: true }),
    ],
  };

  const kimiK3 = {
    sourceLabel: "Moonshot Kimi K3 config + reference forward",
    evidenceLevel: "REFERENCE FORWARD",
    sources: [
      { label: "official config", url: "https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json" },
      { label: "language forward", url: "https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py" },
    ],
    caption: "69 层 KDA 与 24 层 Gated MLA 不会在同一层同时执行。KDA 更新固定大小 recurrent state；MLA 才读 compressed KV。",
    nodes: [
      node("text_ids", 0, "Token IDs", "activation", "Tokenizer output", "[{B}, {T}]", "文本输入索引。", [], { once: true }),
      node("embedding", 0, "Text embedding", "weight", "Lookup", "[163,840 × 7,168]", "映射到 H=7,168。", ["text_ids"], { costId: "embed", once: true, dIn: 163840, dOut: 7168 }),
      node("hidden_in", 0, "Hidden states", "activation", "Embedded sequence", "[{B}, {T}, 7,168]", "decoder 输入。视觉路径不计入文本吞吐。", ["embedding"]),

      node("attn_norm", 1, "RMSNorm", "operation", "Pre-attention norm", "[{B}, {T}, 7,168]", "attention 前归一化。", ["hidden_in"], { costId: "rms", dIn: 7168 }),
      node("kda_qkv", 1, "KDA Q / K / V", "weight", "Three projections", "[{B}, {T}, 96, 128]", "96 × 128 维 QKV。", ["attn_norm"], { layerBranch: "kda", costId: "kda_qkv", dIn: 7168, dOut: 36864 }),
      node("short_conv", 1, "Short conv ×3", "operation", "causal conv1d k=4", "Q/K/V streams", "分别通过 kernel-size 4 的因果卷积。", ["kda_qkv"], { layerBranch: "kda", costId: "kda_conv" }),
      node("kda_gate", 1, "KDA gates", "weight", "full-rank gate + beta", "[{B}, {T}, 96, 128]", "控制 delta update 与遗忘。", ["attn_norm", "short_conv"], { layerBranch: "kda", costId: "kda_gate", dIn: 7168, dOut: 12288 }),
      node("kda_state", 1, "KDA recurrent state", "cache", "delta-rule state", "[{B}, 96, 128, 128]", "不生成 T×T。Decode 只更新固定 state。", ["short_conv", "kda_gate"], { layerBranch: "kda", cacheRole: "readAppend", costId: "attn_kda" }),
      node("kda_output", 1, "KDA output", "operation", "gated RMSNorm → o_proj", "[{B}, {T}, 7,168]", "投影回 H。", ["kda_state"], { layerBranch: "kda", costId: "gemm", dIn: 12288, dOut: 7168 }),
      node("mla_q", 1, "MLA Q latent", "weight", "7,168 → 1,536 → heads", "Qᴺ 128 + Qᴿ 64", "Gated MLA 低秩 query。", ["attn_norm"], { layerBranch: "mla", costId: "mla_q", dIn: 7168, dOut: 1536 }),
      node("mla_kv", 1, "MLA KV latent", "weight", "7,168 → 512+64", "KV latent 512", "compressed KV 与共享 RoPE key。Prefill 写入。", ["attn_norm"], { layerBranch: "mla", cacheRole: "write", costId: "gemm", dIn: 7168, dOut: 576 }),
      node("mla_cache", 1, "MLA KV cache", "cache", "512+64 latent", "[{B}, {S}, 576]", "仅 MLA 层在 Decode 读长序列。", ["mla_kv"], { layerBranch: "mla", cacheRole: "readAppend", costId: "kv_cache_mla" }),
      node("gated_mla", 1, "Gated MLA", "operation", "RoPE attention × gate", "[{B}, {T}, 7,168]", "完整 MLA 再乘 output gate。", ["mla_q", "mla_cache"], { layerBranch: "mla", costId: "attn_gated_mla" }),
      node("attention_output", 1, "Attention output", "activation", "Selected branch", "[{B}, {T}, 7,168]", "当前层类型的 KDA 或 MLA 输出。", ["kda_output", "gated_mla"]),

      node("ffn_norm", 2, "RMSNorm", "operation", "Pre-MoE norm", "[{B}, {T}, 7,168]", "LatentMoE 前归一化。", ["attention_output"], { costId: "rms", dIn: 7168 }),
      node("latent_down", 2, "Latent down", "weight", "Linear 7,168 → 3,584", "[{B}, {T}, 3,584]", "先压到 latent width 再路由。", ["ffn_norm"], { costId: "gemm", dIn: 7168, dOut: 3584 }),
      node("router", 2, "Router", "weight", "sigmoid scoring", "[{B}×{T}, 896]", "896 个 routed expert scores。", ["latent_down"], { costId: "gemm", dIn: 3584, dOut: 896 }),
      node("top16", 2, "Top-16 route", "operation", "select + normalize", "ids [{B}×{T}, 16]", "每 token 激活 16 个专家。", ["router"]),
      node("routed_experts", 2, "896 latent experts", "module", "16 active SiTU", "3,584 → 3,072 → 3,584", "在 latent width 内计算。", ["top16", "latent_down"], { costId: "moe_routed", skipIfEp: true, dIn: 3584, dOut: 3072 }),
      node("shared_experts", 2, "2 shared experts", "module", "Always-on latent MLPs", "3,584 → 2×3,072 → 3,584", "两个 shared experts 始终执行。", ["latent_down"], { costId: "moe_shared", dIn: 3584, dOut: 3072 }),
      node("latent_moe_combine", 2, "LatentMoE combine", "operation", "weighted + shared", "[{B}, {T}, 3,584]", "EP=1 时在此合并。", ["routed_experts", "shared_experts", "top16"], { skipIfEp: true }),

      node("dispatch", 3, "Expert dispatch", "communication", "Sort by expert ID", "[{B}×{T}×16, 3,584]", "按 expert 分组 latent rows。", ["top16", "latent_down"], { epOnly: true }),
      node("all_to_all", 3, "All-to-all", "communication", "EP exchange", "active latent rows", "EP>1 时交换。", ["dispatch"], { epOnly: true, costId: "all_to_all" }),
      node("expert_compute", 3, "Local latent experts", "module", "16 active SiTU", "3,584 → 3,072 → 3,584", "本地专家计算。", ["all_to_all"], { epOnly: true, costId: "moe_routed", dIn: 3584, dOut: 3072 }),
      node("inverse_gather", 3, "Inverse gather", "communication", "Restore order", "[{B}, {T}, 16, 3,584]", "返回原 token 顺序。", ["expert_compute"], { epOnly: true }),
      node("latent_norm", 2, "Latent RMSNorm", "operation", "Normalize mixture", "[{B}, {T}, 3,584]", "up projection 前的 latent norm。", ["latent_moe_combine", "inverse_gather", "shared_experts"], { costId: "rms", dIn: 3584 }),
      node("latent_up", 2, "Latent up", "weight", "Linear 3,584 → 7,168", "[{B}, {T}, 7,168]", "投影回主干宽度。", ["latent_norm"], { costId: "gemm", dIn: 3584, dOut: 7168 }),
      node("attnres_bank", 3, "AttnRes bank", "cache", "every 12th prefix sum", "block states × 7,168", "跨 block 保存 residual。", ["attention_output"], { cacheRole: "readAppend" }),
      node("block_output", 3, "Block output", "activation", "AttnRes + MoE", "[{B}, {T}, 7,168]", "进入下一 decoder layer。", ["latent_up", "attnres_bank"]),

      node("final_norm", 4, "Final RMSNorm", "operation", "Normalize last token", "[{B}, 7,168]", "只用最后位置。", ["block_output"], { costId: "rms", once: true, dIn: 7168 }),
      node("lm_head", 4, "LM head", "weight", "Linear 7,168 → 163,840", "[7,168 × 163,840]", "投影到语言词表。", ["final_norm"], { costId: "gemm", once: true, dIn: 7168, dOut: 163840 }),
      node("logits", 4, "Logits", "activation", "Last-token scores", "[{B}, 163,840]", "采样一个 token。", ["lm_head"], { once: true }),
    ],
  };

  const kimiK25 = {
    sourceLabel: "Moonshot Kimi K2.5 config + reference forward",
    evidenceLevel: "REFERENCE FORWARD",
    sources: [
      { label: "official config", url: "https://huggingface.co/moonshotai/Kimi-K2.5" },
      { label: "language forward", url: "https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/modeling_deepseek.py" },
    ],
    caption: "MLA：Q 走 1,536 latent，KV 走 512+64。Decode 按 serving absorbed cache 计 HBM，不按 HF 展开 K/V。",
    nodes: [
      node("text_ids", 0, "Token IDs", "activation", "Tokenizer output", "[{B}, {T}]", "文本 token 索引。", [], { once: true }),
      node("embedding", 0, "Embedding W", "weight", "Lookup", "[163,840 × 7,168]", "映射到 H=7,168。", ["text_ids"], { costId: "embed", once: true, dIn: 163840, dOut: 7168 }),
      node("hidden_in", 0, "Hidden states", "activation", "Embedded sequence", "[{B}, {T}, 7,168]", "decoder 输入。", ["embedding"]),

      node("attn_norm", 1, "RMSNorm", "operation", "Pre-attention norm", "[{B}, {T}, 7,168]", "epsilon=1e-5。", ["hidden_in"], { costId: "rms", dIn: 7168 }),
      node("q_down", 1, "Q down", "weight", "Linear 7,168 → 1,536", "[7,168 × 1,536]", "q_lora_rank=1,536。", ["attn_norm"], { costId: "gemm", dIn: 7168, dOut: 1536 }),
      node("q_up", 1, "Q up", "weight", "Linear 1,536 → 64×192", "[1,536 × 12,288]", "128 NoPE + 64 RoPE per head。", ["q_down"], { costId: "gemm", dIn: 1536, dOut: 12288 }),
      node("kv_down", 1, "KV down", "weight", "7,168 → 512+64", "[7,168 × 576]", "compressed KV 与共享 RoPE key。", ["attn_norm"], { cacheRole: "write", costId: "gemm", dIn: 7168, dOut: 576 }),
      node("kv_cache", 1, "Absorbed KV cache", "cache", "serving 512+64", "[{B}, {S}, 576]", "HF reference 缓存展开 K/V；本工具 Decode 按 absorbed latent 计。", ["kv_down"], { cacheRole: "readAppend", costId: "kv_cache_mla" }),
      node("kv_up", 1, "KV up", "weight", "512 → 64×256", "[512 × 16,384]", "展开 Kᴺ 与 V。Decode 可吸收进 kernel。", ["kv_cache"], { costId: "gemm", dIn: 512, dOut: 16384 }),
      node("mla_attn", 1, "MLA attention", "operation", "QKᵀ → softmax → V", "[{B}, 64, {T}, {S}]", "组合 NoPE 与 RoPE 子空间。", ["q_up", "kv_up"], { costId: "attn_mla_full" }),
      node("o_proj", 1, "Output Wᴼ", "weight", "8,192 → 7,168", "[8,192 × 7,168]", "多头 context 投影回 H。", ["mla_attn"], { costId: "gemm", dIn: 8192, dOut: 7168 }),
      node("attn_residual", 1, "Attention residual", "activation", "Residual add", "[{B}, {T}, 7,168]", "attention + input。", ["o_proj", "hidden_in"]),

      node("ffn_norm", 2, "RMSNorm", "operation", "Pre-FFN norm", "[{B}, {T}, 7,168]", "MoE 前归一化。", ["attn_residual"], { costId: "rms", dIn: 7168 }),
      node("router", 2, "Router", "weight", "FP32 linear → sigmoid", "[{B}×{T}, 384]", "correction bias 只影响选择。", ["ffn_norm"], { costId: "gemm", dIn: 7168, dOut: 384 }),
      node("topk", 2, "Top-8 route", "operation", "select × 2.827", "ids [{B}×{T}, 8]", "384 中激活 8 个。", ["router"]),
      node("routed_experts", 2, "384 routed experts", "module", "8 active SwiGLU", "7,168 → 2,048 → 7,168", "EP=1 时在此计算。", ["topk", "ffn_norm"], { costId: "moe_routed", skipIfEp: true, dIn: 7168, dOut: 2048 }),
      node("shared_expert", 2, "Shared expert", "module", "Always-on SwiGLU", "7,168 → 2,048 → 7,168", "始终处理 token。", ["ffn_norm"], { costId: "moe_shared", dIn: 7168, dOut: 2048 }),
      node("moe_combine", 2, "MoE combine", "operation", "weighted + shared", "[{B}, {T}, 7,168]", "EP=1 合并路径。", ["routed_experts", "shared_expert", "topk"], { skipIfEp: true }),

      node("dispatch", 3, "Token dispatch", "communication", "argsort expert ids", "[{B}×{T}×8, 7,168]", "按 expert 连续排列。", ["topk", "ffn_norm"], { epOnly: true }),
      node("all_to_all", 3, "All-to-all", "communication", "if EP>1", "sorted token states", "发布 config ep_size=1；分布式可覆盖。", ["dispatch"], { epOnly: true, costId: "all_to_all" }),
      node("expert_compute", 3, "Local experts", "module", "8 active SwiGLU", "7,168 → 2,048 → 7,168", "本地专家。", ["all_to_all"], { epOnly: true, costId: "moe_routed", dIn: 7168, dOut: 2048 }),
      node("gather", 3, "Inverse gather", "communication", "restore order", "[{B}, {T}, 8, 7,168]", "写回 token/expert slot。", ["expert_compute"], { epOnly: true }),
      node("block_residual", 3, "Block residual", "activation", "FFN + residual", "[{B}, {T}, 7,168]", "进入下一层。", ["gather", "moe_combine", "shared_expert", "attn_residual"]),

      node("final_norm", 4, "Final RMSNorm", "operation", "Normalize last token", "[{B}, 7,168]", "只用最后位置。", ["block_residual"], { costId: "rms", once: true, dIn: 7168 }),
      node("lm_head", 4, "LM head", "weight", "7,168 → 163,840", "[7,168 × 163,840]", "投影到词表。", ["final_norm"], { costId: "gemm", once: true, dIn: 7168, dOut: 163840 }),
      node("logits", 4, "Logits", "activation", "Last-token scores", "[{B}, 163,840]", "采样一个 token。", ["lm_head"], { once: true }),
    ],
  };

  const minimaxM3 = {
    sourceLabel: "MiniMax M3 official config + vLLM serving forward",
    evidenceLevel: "SERVING FORWARD",
    sources: [
      { label: "official config", url: "https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json" },
      { label: "vLLM sparse attention", url: "https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/common/sparse_attention/" },
    ],
    caption: "代表层是 sparse GQA：indexer 选 16 个 128-token blocks + local block。KV 仍按全量 paged 占显存，Attention 只读被选块。",
    nodes: [
      node("text_ids", 0, "Token IDs", "activation", "Tokenizer output", "[{B}, {T}]", "文本 token 索引。", [], { once: true }),
      node("embedding", 0, "Text embedding", "weight", "Lookup", "[200,064 × 6,144]", "映射到 H=6,144。", ["text_ids"], { costId: "embed", once: true, dIn: 200064, dOut: 6144 }),
      node("hidden_in", 0, "Hidden states", "activation", "Embedded sequence", "[{B}, {T}, 6,144]", "decoder 输入。", ["embedding"]),

      node("attn_norm", 1, "Gemma RMSNorm", "operation", "Pre-attention norm", "[{B}, {T}, 6,144]", "Gemma-style RMSNorm。", ["hidden_in"], { costId: "rms", dIn: 6144 }),
      node("q_proj", 1, "Q projection", "weight", "64 heads × 128", "[{B}, {T}, 64, 128]", "主 attention queries。", ["attn_norm"], { costId: "gemm", dIn: 6144, dOut: 8192 }),
      node("kv_proj", 1, "K / V projections", "weight", "4 KV heads × 128", "[{B}, {T}, 4, 128]", "GQA：每 16 个 Q 共享 1 组 KV。Prefill 写入 paged cache。", ["attn_norm"], { cacheRole: "write", costId: "gemm", dIn: 6144, dOut: 1024 }),
      node("kv_cache", 1, "Paged KV cache", "cache", "full sequence storage", "[{B}, {S}, 4, 256]", "显存按全量 S 计；Decode 计算只读选中 blocks。", ["kv_proj"], { cacheRole: "readAppend", costId: "kv_cache_gqa" }),
      node("indexer", 1, "Lightning indexer", "module", "4 heads × 128", "block scores", "对 KV blocks 打分。", ["attn_norm"], { costId: "m3_indexer" }),
      node("index_topk", 1, "Top-16 KV blocks", "operation", "select + local block", "16 × 128 + local", "稀疏单位是 block 不是单个 token。", ["indexer"]),
      node("sparse_gqa", 1, "Sparse GQA", "operation", "attend selected blocks", "[{B}, 64, {T}, 128]", "读取 indexer 选择的 paged KV。", ["q_proj", "kv_cache", "index_topk"], { costId: "attn_sparse_gqa" }),
      node("attn_out", 1, "Output projection", "weight", "64×128 → 6,144", "[8,192 × 6,144]", "投影回 H。", ["sparse_gqa"], { costId: "gemm", dIn: 8192, dOut: 6144 }),
      node("attn_residual", 1, "Attention residual", "activation", "Residual add", "[{B}, {T}, 6,144]", "attention + input。", ["attn_out", "hidden_in"]),

      node("ffn_norm", 2, "Gemma RMSNorm", "operation", "Pre-FFN norm", "[{B}, {T}, 6,144]", "MoE 前归一化。", ["attn_residual"], { costId: "rms", dIn: 6144 }),
      node("router", 2, "Router", "weight", "linear → sigmoid", "[{B}×{T}, 128]", "routed expert scores。", ["ffn_norm"], { costId: "gemm", dIn: 6144, dOut: 128 }),
      node("top4", 2, "Top-4 route", "operation", "select ×2", "ids [{B}×{T}, 4]", "128 中激活 4 个。", ["router"]),
      node("routed_experts", 2, "128 routed experts", "module", "4 active SwiGLU-OAI", "6,144 → 3,072 → 6,144", "EP=1 时在此计算。", ["top4", "ffn_norm"], { costId: "moe_routed", skipIfEp: true, dIn: 6144, dOut: 3072 }),
      node("shared_expert", 2, "Shared expert", "module", "Always-on SwiGLU-OAI", "6,144 → 3,072 → 6,144", "始终激活。", ["ffn_norm"], { costId: "moe_shared", dIn: 6144, dOut: 3072 }),
      node("moe_combine", 2, "MoE combine", "operation", "weighted + shared", "[{B}, {T}, 6,144]", "EP=1 合并路径。", ["routed_experts", "shared_expert", "top4"], { skipIfEp: true }),

      node("dispatch", 3, "Expert dispatch", "communication", "Group Top-4 rows", "[{B}×{T}×4, 6,144]", "按 expert 分组。", ["top4", "ffn_norm"], { epOnly: true }),
      node("all_to_all", 3, "All-to-all", "communication", "EP exchange", "active token rows", "专家跨设备时发生。", ["dispatch"], { epOnly: true, costId: "all_to_all" }),
      node("expert_compute", 3, "Local experts", "module", "4 active SwiGLU", "6,144 → 3,072 → 6,144", "本地专家。", ["all_to_all"], { epOnly: true, costId: "moe_routed", dIn: 6144, dOut: 3072 }),
      node("inverse_gather", 3, "Inverse gather", "communication", "restore order", "[{B}, {T}, 4, 6,144]", "恢复 token 顺序。", ["expert_compute"], { epOnly: true }),
      node("block_residual", 3, "Block residual", "activation", "FFN + residual", "[{B}, {T}, 6,144]", "进入下一层。", ["inverse_gather", "moe_combine", "shared_expert", "attn_residual"]),

      node("final_norm", 4, "Final Gemma RMSNorm", "operation", "Normalize last token", "[{B}, 6,144]", "只用最后位置。", ["block_residual"], { costId: "rms", once: true, dIn: 6144 }),
      node("lm_head", 4, "LM head", "weight", "6,144 → 200,064", "[6,144 × 200,064]", "投影到词表。", ["final_norm"], { costId: "gemm", once: true, dIn: 6144, dOut: 200064 }),
      node("logits", 4, "Logits", "activation", "Last-token scores", "[{B}, 200,064]", "采样一个 token。", ["lm_head"], { once: true }),
    ],
  };

  window.MODEL_TENSOR_GRAPHS = {
    "glm-52": glm52,
    "kimi-k3": kimiK3,
    "kimi-k25": kimiK25,
    "minimax-m3": minimaxM3,
  };
}());
