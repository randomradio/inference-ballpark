(function () {
  const zero = () => ({ flops: 0, flopsEq: 0, bytesRead: 0, bytesWrite: 0, comm: 0, commIntra: 0 });
  const pack = (c) => ({
    flops: c.flops || 0,
    // FP8 当量 FLOPs：BF16/INT4-marlin 的 GEMM 只能吃到一半 tensor core 峰值
    flopsEq: c.flopsEq ?? (c.flops || 0),
    bytesRead: c.bytesRead || 0,
    bytesWrite: c.bytesWrite || 0,
    comm: c.comm || 0,
    commIntra: c.commIntra || 0,
  });
  const add = (a, b) => {
    a = pack(a); b = pack(b);
    return {
      flops: a.flops + b.flops,
      flopsEq: a.flopsEq + b.flopsEq,
      bytesRead: a.bytesRead + b.bytesRead,
      bytesWrite: a.bytesWrite + b.bytesWrite,
      comm: a.comm + b.comm,
      commIntra: a.commIntra + b.commIntra,
    };
  };
  const scale = (c, n) => {
    c = pack(c);
    return {
      flops: c.flops * n,
      flopsEq: c.flopsEq * n,
      bytesRead: c.bytesRead * n,
      bytesWrite: c.bytesWrite * n,
      comm: c.comm * n,
      commIntra: c.commIntra * n,
    };
  };
  // 把各 dtype 的 FLOPs 折成「FP8 峰值当量」：T = flopsEq / (FP8_peak × MFU)。
  // Hopper：无原生 FP4，MXFP4/INT4 走 marlin → 按 BF16（默认 FP8/2）。
  // Rubin：MXFP4 吃 NVFP4 峰值（50 PFLOPS），BF16 吃 4 PFLOPS，FP8 吃 17.5 PFLOPS。
  function flopScale(bytes, ctx) {
    const peaks = ctx?.peaks || {};
    const fp8 = peaks.fp8 || 0;
    const b = bytes ?? 1;
    if (b <= 0.5) {
      if (peaks.fp4 > 0 && fp8 > 0) return fp8 / peaks.fp4;
      if (peaks.bf16 > 0 && fp8 > 0) return fp8 / peaks.bf16;
      return 2;
    }
    if (b >= 2) {
      if (peaks.bf16 > 0 && fp8 > 0) return fp8 / peaks.bf16;
      return 2;
    }
    return 1;
  }

  function replicaGpus(hw) {
    const pp = Math.max(1, hw.pp || 1);
    if (hw.layout === "mesh") return Math.max(1, hw.tp * hw.ep * pp);
    return Math.max(1, hw.tp || 1, hw.ep || 1) * pp;
  }

  function commPlan(hw) {
    const gpn = Math.max(1, hw.gpn || 8);
    const replica = replicaGpus(hw);
    const world = replica / Math.max(1, hw.pp || 1);
    const replicaNodes = Math.max(1, Math.ceil(replica / gpn));
    const layout = hw.layout || "overlap";
    const attnMode = hw.attnMode || (layout === "overlap" ? "dp" : "tp");
    if (layout === "overlap") {
      const onNode = world <= gpn;
      const fabric = onNode ? "nvlink" : "roce";
      const epFabric = onNode && hw.ep <= gpn ? "nvlink" : "roce";
      const attn = attnMode === "tp" ? "tp" : "dp";
      return {
        replica,
        world,
        replicaNodes,
        attn,
        tpFabric: fabric,
        epFabric,
        note: onNode
          ? (attn === "dp"
            ? `replica ${replica} 卡 / ${replicaNodes} 节点 · TP=${hw.tp} 切 dense/MLA · EP=${hw.ep} DeepEP · Attention DP 切 KV · 节点内 NVLink`
            : `replica ${replica} 卡 / ${replicaNodes} 节点 · TP=${hw.tp}∩EP=${hw.ep} · Attention 走 TP（KDA 为主）· 节点内 NVLink`)
          : (attn === "dp"
            ? `replica ${replica} 卡、占 ${replicaNodes} 个节点 · 跨节点 A2A / AR 走 RoCE`
            : `replica ${replica} 卡、占 ${replicaNodes} 个节点 · TP=${hw.tp}∩EP=${hw.ep} · Attention TP · 跨节点 A2A/AR 走 RoCE`),
      };
    }
    const tpFabric = hw.tp <= gpn ? "nvlink" : "roce";
    const epOnNode = hw.ep <= 1 || hw.tp * hw.ep <= gpn;
    const epFabric = epOnNode ? "nvlink" : "roce";
    return {
      replica,
      world: replica / Math.max(1, hw.pp || 1),
      replicaNodes,
      attn: "tp",
      tpFabric,
      epFabric,
      note: `网格 TP=${hw.tp}→${tpFabric} · EP=${hw.ep}→${epFabric} · replica ${replica} 卡 / ${replicaNodes} 节点`,
    };
  }

  function tpN(ctx) { return Math.max(1, ctx.tp || 1); }
  function epN(ctx) { return Math.max(1, ctx.ep || 1); }
  function worldN(ctx) { return Math.max(tpN(ctx), epN(ctx)); }
  function isOverlap(ctx) { return (ctx.layout || "overlap") !== "mesh"; }
  function isDpAttn(ctx) { return isOverlap(ctx) && (ctx.attnMode || "dp") !== "tp"; }

  function flopShard(ctx, shard) {
    if (shard === "none") return 1;
    if (isOverlap(ctx)) {
      if (shard === "ep" || shard === "tp-ep") return epN(ctx);
      return isDpAttn(ctx) ? worldN(ctx) : tpN(ctx);
    }
    if (shard === "tp") return tpN(ctx);
    if (shard === "ep") return epN(ctx);
    if (shard === "tp-ep") return tpN(ctx) * epN(ctx);
    return 1;
  }

  function weightShard(ctx, shard) {
    if (shard === "none") return 1;
    if (shard === "ep") return epN(ctx);
    if (shard === "tp-ep") return isOverlap(ctx) ? epN(ctx) : tpN(ctx) * epN(ctx);
    return tpN(ctx);
  }

  function localB(ctx) {
    return isDpAttn(ctx) ? ctx.B / worldN(ctx) : ctx.B;
  }

  function gemm(ctx, dIn, dOut, shard = "tp", wBytes) {
    const { B, T, aBytes } = ctx;
    const p = flopShard(ctx, shard);
    const w = weightShard(ctx, shard);
    const wb = wBytes ?? ctx.wBytes;
    const flops = 2 * B * T * dIn * dOut / p;
    return {
      flops,
      flopsEq: flops * flopScale(wb, ctx),
      bytesRead: dIn * dOut * wb / w + B * T * dIn * aBytes / p,
      bytesWrite: (B * T * dOut * aBytes) / p,
      comm: 0,
    };
  }

  function rms(ctx, dim) {
    const p = isDpAttn(ctx) ? worldN(ctx) : (isOverlap(ctx) ? tpN(ctx) : 1);
    return {
      flops: 4 * ctx.B * ctx.T * dim / p,
      bytesRead: ctx.B * ctx.T * dim * ctx.aBytes / p,
      bytesWrite: ctx.B * ctx.T * dim * ctx.aBytes / p,
      comm: 0,
    };
  }

  function swiglu(ctx, h, hidden, nExperts, shard = "tp") {
    const expertSplit = shard === "tp-ep" || shard === "ep";
    const local = nExperts / (expertSplit ? epN(ctx) : 1);
    const gShard = expertSplit && isOverlap(ctx) ? "none" : "tp";
    const bytes = expertSplit ? ctx.wBytes : (ctx.denseWBytes || ctx.attnWBytes || ctx.wBytes);
    const c = scale(add(gemm(ctx, h, hidden * 2, gShard, bytes), gemm(ctx, hidden, h, gShard, bytes)), local);
    if (expertSplit) {
      // 权重读要按本卡实际被命中的 distinct expert 数整块计：
      // 小 batch decode 每步仍要把每个被路由到的 expert 权重完整读一次，
      // 不能按 token 份额摊薄（这是小 batch MoE decode 吞吐差的物理原因）。
      const perExpertW = 3 * h * hidden * bytes / weightShard(ctx, gShard);
      const resident = Math.max(1, (ctx.dims.experts || nExperts) / epN(ctx));
      const hits = (ctx.B * ctx.T * nExperts) / epN(ctx);
      const distinct = Math.min(resident, hits);
      c.bytesRead += Math.max(0, distinct - local) * perExpertW;
    }
    return c;
  }

  function attnQKAV(ctx, heads, tq, tk, dHead) {
    // K/V 与 cache 的 HBM 流量由 kv_cache_* / 状态节点单独计
    // （MLA latent / GQA KV / indexer K 都是 head 共享的，不能按 head 数重复读）。
    // 这里只计 QK^T + AV 的 FLOPs 和 Q/输出激活。
    const { aBytes } = ctx;
    const B = localB(ctx);
    const h = isDpAttn(ctx) ? heads : heads / tpN(ctx);
    const flops = 4 * B * h * tq * tk * dHead;
    const bytesRead = B * h * tq * dHead * aBytes;
    const bytesWrite = B * h * tq * dHead * aBytes;
    return { flops, flopsEq: flops * flopScale(ctx.attnWBytes ?? 1, ctx), bytesRead, bytesWrite, comm: 0 };
  }

  function placeBytes(ctx, bytes, fabric) {
    if (!bytes) return zero();
    if (fabric === "roce") return { flops: 0, bytesRead: 0, bytesWrite: 0, comm: bytes };
    return { flops: 0, bytesRead: 0, bytesWrite: 0, comm: 0, commIntra: bytes };
  }

  function allToAll(ctx, h, topk) {
    const tokens = (ctx.B * ctx.T * topk) / epN(ctx);
    const bytes = 2 * tokens * h * ctx.aBytes;
    return placeBytes(ctx, bytes, ctx.epFabric || (epN(ctx) <= (ctx.gpn || 8) ? "nvlink" : "roce"));
  }

  function tpAllReduce(ctx, h) {
    if (isDpAttn(ctx) || tpN(ctx) <= 1) return zero();
    const bytes = 2 * ctx.B * ctx.T * h * ctx.aBytes;
    return placeBytes(ctx, bytes, ctx.tpFabric || "nvlink");
  }

  function nodeCost(node, ctx) {
    const d = ctx.dims;
    switch (node.costId) {
      case "gemm":
        return gemm(ctx, node.dIn, node.dOut, "tp");
      case "embed": {
        // 查表只读 B×T 行，不是整张 embedding 表
        const p = isDpAttn(ctx) ? worldN(ctx) : tpN(ctx);
        const wb = ctx.denseWBytes || ctx.wBytes;
        return {
          flops: 0,
          bytesRead: ctx.B * ctx.T * d.H * wb / p,
          bytesWrite: ctx.B * ctx.T * d.H * ctx.aBytes / p,
          comm: 0,
        };
      }
      case "rms":
        return rms(ctx, node.dIn || d.H);
      case "rope":
        return { flops: 8 * ctx.B * ctx.T * (d.heads || 64) * (d.qkRope || 64) / tpN(ctx), bytesRead: 0, bytesWrite: 0, comm: 0 };
      case "indexer_q":
        return gemm(ctx, node.dIn || d.qLatent, (d.indexHeads || 32) * (d.indexDim || 128), "tp");
      case "indexer_k":
        return gemm(ctx, d.H, d.indexDim || 128, "tp");
      case "indexer_scan": {
        const heads = d.indexHeads || 32;
        const dim = d.indexDim || 128;
        const scan = attnQKAV(ctx, heads, ctx.T, ctx.S, dim / 2);
        // indexer K cache 是 head 共享的一份 [S, dim]，整段读一次
        scan.bytesRead += localB(ctx) * ctx.S * dim * ctx.kvBytes;
        return scan;
      }
      case "attn_sparse_mla":
        return attnQKAV(ctx, d.heads, ctx.T, Math.min(ctx.S, d.indexTopK || 2048), d.valueHead || 256);
      case "kv_cache_mla": {
        const width = (d.kvLatent || 512) + (d.qkRope || 64);
        const B = localB(ctx);
        const read = ctx.mode === "decode";
        // DSA 稀疏注意力 decode 只 gather indexer 选出的 top-K 行
        const readS = d.indexHeads && d.indexTopK ? Math.min(ctx.S, d.indexTopK) : ctx.S;
        const bytes = B * (read ? readS : ctx.T) * width * ctx.kvBytes;
        return { flops: 0, bytesRead: read ? bytes : 0, bytesWrite: read ? B * width * ctx.kvBytes : bytes, comm: 0 };
      }
      case "kv_cache_gqa": {
        const width = (d.kvHeads || 4) * (d.headDim || 128) * 2;
        const B = isDpAttn(ctx) ? localB(ctx) : ctx.B;
        const shard = isDpAttn(ctx) ? 1 : Math.min(tpN(ctx), d.kvHeads || 4);
        const read = ctx.mode === "decode";
        const selected = (d.indexTopK + (d.localBlocks || 1)) * (d.blockSize || 128);
        const traffic = B * Math.min(ctx.S, selected) * width * ctx.kvBytes / shard;
        const write = B * (read ? 1 : ctx.T) * width * ctx.kvBytes / shard;
        return { flops: 0, bytesRead: read ? traffic : 0, bytesWrite: write, comm: 0 };
      }
      case "kda_qkv":
        return gemm(ctx, d.H, 3 * d.heads * d.headDim, "tp");
      case "kda_conv": {
        const B = localB(ctx);
        const heads = isDpAttn(ctx) ? d.heads : d.heads / tpN(ctx);
        return {
          flops: 3 * B * ctx.T * heads * d.headDim * 4,
          bytesRead: B * ctx.T * heads * d.headDim * 3 * ctx.aBytes,
          bytesWrite: 0,
          comm: 0,
        };
      }
      case "kda_gate":
        return gemm(ctx, d.H, d.heads * d.headDim, "tp");
      case "attn_kda": {
        const B = localB(ctx);
        const heads = isDpAttn(ctx) ? d.heads : d.heads / tpN(ctx);
        const state = B * heads * d.headDim * d.headDim * ctx.kvBytes;
        return {
          flops: 2 * B * ctx.T * heads * d.headDim * d.headDim,
          bytesRead: state + B * ctx.T * heads * d.headDim * ctx.aBytes,
          bytesWrite: state,
          comm: 0,
        };
      }
      case "mla_q":
        return add(
          gemm(ctx, d.H, d.qLatent, "tp"),
          gemm(ctx, d.qLatent, d.heads * ((d.qkNope || 128) + (d.qkRope || 64)), "tp"),
        );
      case "attn_gated_mla":
        return attnQKAV(ctx, d.heads, ctx.T, ctx.S, (d.qkNope || 128) + (d.qkRope || 64));
      case "attn_mla_full":
        return attnQKAV(ctx, d.heads, ctx.T, ctx.S, d.valueHead || 128);
      case "m3_indexer":
        return attnQKAV(ctx, 4, ctx.T, Math.ceil(ctx.S / (d.blockSize || 128)), d.headDim || 128);
      case "attn_sparse_gqa": {
        const tk = Math.min(ctx.S, ((d.indexTopK || 16) + (d.localBlocks || 1)) * (d.blockSize || 128));
        return attnQKAV(ctx, d.heads, ctx.T, tk, d.headDim || 128);
      }
      case "moe_routed":
        return swiglu(ctx, node.dIn || d.H, node.dOut || d.expertHidden, d.activeExperts, "tp-ep");
      case "moe_shared":
        return swiglu(ctx, node.dIn || d.H, node.dOut || d.expertHidden, d.sharedExperts || 1, "tp");
      case "all_to_all":
        return allToAll(ctx, node.dIn || d.latent || d.H, d.activeExperts);
      default:
        return zero();
    }
  }

  function layerAttention(model, ctx, kind) {
    ctx = { ...ctx, wBytes: ctx.attnWBytes || ctx.wBytes };
    const d = model.dims;
    if (kind === "dsa-mla") {
      let c = zero();
      c = add(c, rms(ctx, d.H));
      c = add(c, gemm(ctx, d.H, d.qLatent));
      c = add(c, gemm(ctx, d.qLatent, d.heads * (d.qkNope + d.qkRope)));
      c = add(c, gemm(ctx, d.H, d.kvLatent + d.qkRope));
      c = add(c, gemm(ctx, d.kvLatent, d.heads * (d.qkNope + d.valueHead)));
      c = add(c, gemm(ctx, d.heads * d.valueHead, d.H));
      const indexer = ctx.indexShareSkip ? zero() : add(
        gemm(ctx, d.qLatent, d.indexHeads * d.indexDim),
        add(gemm(ctx, d.H, d.indexDim), nodeCost({ costId: "indexer_scan" }, ctx)),
      );
      c = add(c, indexer);
      c = add(c, attnQKAV(ctx, d.heads, ctx.T, Math.min(ctx.S, d.indexTopK), d.valueHead));
      c = add(c, nodeCost({ costId: "kv_cache_mla" }, ctx));
      c = add(c, tpAllReduce(ctx, d.H));
      return c;
    }
    if (kind === "kda") {
      let c = rms(ctx, d.H);
      c = add(c, gemm(ctx, d.H, 3 * d.heads * d.headDim));
      c = add(c, nodeCost({ costId: "kda_conv" }, ctx));
      c = add(c, nodeCost({ costId: "attn_kda" }, ctx));
      c = add(c, gemm(ctx, d.heads * d.headDim, d.H));
      c = add(c, tpAllReduce(ctx, d.H));
      return c;
    }
    if (kind === "gated-mla" || kind === "mla") {
      let c = rms(ctx, d.H);
      c = add(c, gemm(ctx, d.H, d.qLatent));
      c = add(c, gemm(ctx, d.qLatent, d.heads * ((d.qkNope || 128) + (d.qkRope || 64))));
      c = add(c, gemm(ctx, d.H, d.kvLatent + d.qkRope));
      const dHead = d.valueHead || ((d.qkNope || 128) + (d.qkRope || 64));
      c = add(c, gemm(ctx, d.kvLatent, d.heads * dHead));
      c = add(c, attnQKAV(ctx, d.heads, ctx.T, ctx.S, dHead));
      c = add(c, gemm(ctx, d.heads * (d.valueHead || d.headDim || 128), d.H));
      c = add(c, nodeCost({ costId: "kv_cache_mla" }, ctx));
      c = add(c, tpAllReduce(ctx, d.H));
      return c;
    }
    if (kind === "sparse-gqa") {
      let c = rms(ctx, d.H);
      c = add(c, gemm(ctx, d.H, d.heads * d.headDim));
      c = add(c, gemm(ctx, d.H, d.kvHeads * d.headDim * 2));
      c = add(c, nodeCost({ costId: "m3_indexer" }, ctx));
      c = add(c, nodeCost({ costId: "attn_sparse_gqa" }, ctx));
      c = add(c, gemm(ctx, d.heads * d.headDim, d.H));
      c = add(c, nodeCost({ costId: "kv_cache_gqa" }, ctx));
      c = add(c, tpAllReduce(ctx, d.H));
      return c;
    }
    if (kind === "dense-gqa") {
      let c = rms(ctx, d.H);
      c = add(c, gemm(ctx, d.H, d.heads * d.headDim));
      c = add(c, gemm(ctx, d.H, d.kvHeads * d.headDim * 2));
      c = add(c, attnQKAV(ctx, d.heads, ctx.T, ctx.S, d.headDim));
      c = add(c, gemm(ctx, d.heads * d.headDim, d.H));
      c = add(c, nodeCost({ costId: "kv_cache_gqa" }, ctx));
      c = add(c, tpAllReduce(ctx, d.H));
      return c;
    }
    return zero();
  }

  function layerFfn(model, ctx, kind) {
    const d = model.dims;
    const h = kind === "latent-moe" ? d.latent : d.H;
    let c = rms(ctx, d.H);
    if (kind === "latent-moe") {
      c = add(c, gemm(ctx, d.H, d.latent));
      c = add(c, gemm(ctx, d.latent, d.experts, "none"));
      c = add(c, swiglu(ctx, d.latent, d.expertHidden, d.activeExperts, "tp-ep"));
      c = add(c, swiglu(ctx, d.latent, d.expertHidden, d.sharedExperts, "tp"));
      c = add(c, gemm(ctx, d.latent, d.H));
    } else if (kind === "moe") {
      c = add(c, gemm(ctx, d.H, d.experts, "none"));
      c = add(c, swiglu(ctx, d.H, d.expertHidden, d.activeExperts, "tp-ep"));
      c = add(c, swiglu(ctx, d.H, d.expertHidden, d.sharedExperts || 1, "tp"));
    } else if (kind === "dense") {
      const hidden = d.denseHidden || d.expertHidden * 4;
      c = add(c, swiglu(ctx, d.H, hidden, 1, "tp"));
    }
    if (epN(ctx) > 1 && kind !== "dense") c = add(c, allToAll(ctx, h, d.activeExperts));
    c = add(c, tpAllReduce(ctx, d.H));
    return c;
  }

  function stackCost(model, ctx) {
    const d = model.dims;
    const attn = model.layerKinds?.attention;
    const ffn = model.layerKinds?.ffn;
    let c = zero();
    if (attn === "dsa-mla") {
      const full = { ...ctx, indexShareSkip: false };
      const share = { ...ctx, indexShareSkip: true };
      const group = d.indexShareGroup || 4;
      const fullN = Math.ceil(d.layers / group);
      const shareN = d.layers - fullN;
      c = add(c, scale(layerAttention(model, full, "dsa-mla"), fullN));
      c = add(c, scale(layerAttention(model, share, "dsa-mla"), shareN));
    } else if (attn === "kda-or-mla") {
      c = add(c, scale(layerAttention(model, ctx, "kda"), d.kdaLayers));
      c = add(c, scale(layerAttention(model, ctx, "gated-mla"), d.mlaLayers));
    } else if (attn === "mla") {
      c = add(c, scale(layerAttention(model, ctx, "mla"), d.layers));
    } else {
      c = add(c, scale(layerAttention(model, ctx, "dense-gqa"), d.denseLayers));
      c = add(c, scale(layerAttention(model, ctx, "sparse-gqa"), d.moeLayers));
    }
    if (ffn === "latent-moe") {
      c = add(c, scale(layerFfn(model, ctx, "dense"), d.denseLayers || 0));
      c = add(c, scale(layerFfn(model, ctx, "latent-moe"), d.moeLayers || d.layers));
    } else {
      c = add(c, scale(layerFfn(model, ctx, "dense"), d.denseLayers || 0));
      c = add(c, scale(layerFfn(model, ctx, "moe"), d.moeLayers || 0));
    }
    return c;
  }

  function onceCost(model, ctx) {
    const d = model.dims;
    // embedding 按整段 T 查表；final norm 与 lm_head 只算最后一个 token
    const last = { ...ctx, T: 1, wBytes: ctx.denseWBytes || ctx.attnWBytes || ctx.wBytes };
    return add(add(nodeCost({ costId: "embed" }, ctx), rms(last, d.H)), gemm(last, d.H, d.vocabulary, "tp"));
  }

  function timeOf(cost, hw) {
    cost = pack(cost);
    const Tc = cost.flopsEq / hw.flopsPerGpu;
    const Tm = (cost.bytesRead + cost.bytesWrite) / hw.hbmPerGpu;
    const intra = hw.intraPerGpu || hw.hbmPerGpu * 0.18;
    const TnNet = cost.comm / Math.max(hw.netPerGpu, 1e-9);
    const TnIntra = cost.commIntra / Math.max(intra, 1e-9);
    const Tn = Math.max(TnNet, TnIntra);
    const T = Math.max(Tc, Tm, Tn);
    const bound = Tc >= Tm && Tc >= Tn ? "计算" : Tm >= Tn ? "HBM" : TnNet >= TnIntra ? "网络" : "节点内";
    return { Tc, Tm, Tn, T, bound };
  }

  function mlaAttnParams(d) {
    const qUp = d.heads * ((d.qkNope || 128) + (d.qkRope || 64));
    const kvUp = d.heads * ((d.qkNope || 128) + (d.valueHead || 128));
    const oIn = d.heads * (d.valueHead || 128);
    return d.H * d.qLatent + d.qLatent * qUp + d.H * (d.kvLatent + d.qkRope) + d.kvLatent * kvUp + oIn * d.H;
  }

  function weightBytesPerGpu(model, tp, ep, pp) {
    const d = model.dims;
    const routedB = d.weightBytes;
    const attnB = d.attnBytes ?? routedB;
    const denseB = d.denseBytes ?? attnB;
    const dense = (h, hidden, bytes) => 3 * h * hidden * bytes;
    const tpDiv = Math.max(1, tp || 1);
    const epDiv = Math.max(1, ep || 1);
    const ppDiv = Math.max(1, pp || 1);
    const attnKind = model.layerKinds?.attention;
    const ffn = model.layerKinds?.ffn;
    let attn = 0;
    if (attnKind === "dsa-mla") {
      attn = (d.H * d.qLatent + d.qLatent * d.heads * (d.qkNope + d.qkRope) + d.H * (d.kvLatent + d.qkRope) + d.kvLatent * d.heads * (d.qkNope + d.valueHead) + d.heads * d.valueHead * d.H) * attnB * d.layers;
    } else if (attnKind === "kda-or-mla") {
      const kda = (3 * d.H * d.heads * d.headDim + d.heads * d.headDim * d.H + d.H * d.heads * d.headDim) * attnB;
      const mla = mlaAttnParams(d) * attnB;
      attn = kda * (d.kdaLayers || 0) + mla * (d.mlaLayers || 0);
    } else if (attnKind === "mla") {
      attn = mlaAttnParams(d) * attnB * d.layers;
    } else {
      attn = (d.H * d.heads * d.headDim + d.H * (d.kvHeads || 4) * (d.headDim || 128) * 2 + d.heads * d.headDim * d.H) * attnB * d.layers;
    }
    let proj = 0;
    let routed = 0;
    let shared = 0;
    const moeLayers = d.moeLayers || d.layers;
    if (ffn === "latent-moe") {
      proj = (d.H * d.latent + d.latent * d.H) * routedB;
      routed = (d.experts / epDiv) * dense(d.latent, d.expertHidden, routedB);
      shared = (d.sharedExperts || 0) * dense(d.latent, d.expertHidden, denseB);
    } else {
      routed = ((d.experts || 1) / epDiv) * dense(d.H, d.expertHidden, routedB);
      shared = dense(d.H, d.expertHidden, denseB);
    }
    const denseLayers = d.denseLayers || 0;
    const denseFfn = denseLayers
      ? dense(d.H, d.denseHidden || d.expertHidden * 4, denseB) * denseLayers / tpDiv
      : 0;
    const ffnLayers = ffn === "latent-moe" ? moeLayers : moeLayers;
    const stack = attn / tpDiv
      + (proj / tpDiv + routed + shared / tpDiv) * ffnLayers
      + denseFfn;
    const embed = (d.vocabulary * d.H * 2) * denseB / tpDiv;
    return stack / ppDiv + embed;
  }

  function kvBytesPerSequence(model, S, B, tp) {
    const d = model.dims;
    const k = d.kvBytes;
    const kind = model.cacheKind;
    if (kind === "hybrid-kda-mla") {
      const state = B * d.kdaLayers * (d.heads / Math.max(1, tp)) * d.headDim * d.headDim * k;
      const mla = B * S * d.mlaLayers * (d.kvLatent + d.qkRope) * k;
      return state + mla;
    }
    if (kind === "paged-gqa") {
      const shard = Math.min(Math.max(1, tp || 1), d.kvHeads || 4);
      return B * S * d.layers * d.kvHeads * d.headDim * 2 * k / shard;
    }
    const indexer = d.indexDim || 0;
    return B * S * d.layers * ((d.kvLatent || 512) + (d.qkRope || 64) + indexer) * k;
  }

  function makeCtx(model, hw, req, mode) {
    const Tmiss = Math.max(1, Math.round(req.input * (1 - req.cache)));
    const T = mode === "prefill" ? Tmiss : 1;
    const S = req.input;
    const plan = commPlan(hw);
    return {
      mode,
      B: req.batch,
      T,
      S,
      dims: model.dims,
      wBytes: model.dims.weightBytes,
      attnWBytes: model.dims.attnBytes ?? model.dims.weightBytes,
      denseWBytes: model.dims.denseBytes ?? model.dims.attnBytes ?? model.dims.weightBytes,
      aBytes: model.dims.kvBytes,
      kvBytes: model.dims.kvBytes,
      tp: hw.tp,
      ep: hw.ep,
      pp: hw.pp,
      gpn: hw.gpn,
      layout: hw.layout || "overlap",
      attnMode: hw.attnMode || (hw.layout === "mesh" ? "tp" : "dp"),
      tpFabric: plan.tpFabric,
      epFabric: plan.epFabric,
      cacheKind: model.cacheKind,
      peaks: hw.peaks || {},
    };
  }

  function estimate(model, hw, req) {
    const gpus = hw.nodes * hw.gpn;
    const perReplica = replicaGpus(hw);
    const plan = commPlan(hw);
    const weights = weightBytesPerGpu(model, hw.tp, hw.ep, hw.pp);
    const usable = Math.max(0, hw.hbmCapBytes * 0.88 - weights);
    const perSeq = kvBytesPerSequence(model, req.input, 1, plan.attn === "dp" ? 1 : hw.tp);
    const kvMaxPerGpu = perSeq > 0 ? Math.max(1, Math.floor(usable / perSeq)) : req.batch;
    const kvMaxB = plan.attn === "dp" ? kvMaxPerGpu * Math.max(1, plan.world) : kvMaxPerGpu;
    const effectiveB = Math.min(req.batch, kvMaxB);
    const kvLimited = effectiveB < req.batch;
    const reqB = { ...req, batch: effectiveB };

    const prefillCtx = makeCtx(model, hw, reqB, "prefill");
    const decodeCtx = makeCtx(model, hw, reqB, "decode");
    const prefillCost = add(stackCost(model, prefillCtx), onceCost(model, prefillCtx));
    const decodeCost = add(stackCost(model, decodeCtx), onceCost(model, { ...decodeCtx, T: 1 }));
    const prefill = timeOf(prefillCost, hw);
    const decode = timeOf(decodeCost, hw);
    const tpot = decode.T / Math.max(req.mtpAccept, 0.25);

    const fits = gpus >= perReplica;
    const replicasTotal = fits ? Math.floor(gpus / perReplica) : 0;

    let clusterToks = 0;
    let servingLabel = "聚合";
    if (!fits) {
      servingLabel = `装不下 replica ${perReplica}`;
    } else if (hw.servingMode === "disaggregated" && hw.pdP + hw.pdD > 0) {
      const share = hw.pdP / (hw.pdP + hw.pdD);
      const gpusP = Math.max(perReplica, Math.floor(gpus * share / perReplica) * perReplica);
      const gpusD = Math.max(perReplica, gpus - gpusP);
      const poolFits = gpusP >= perReplica && gpusD >= perReplica && gpusP + gpusD <= gpus;
      if (!poolFits) {
        servingLabel = `P:D 装不下 replica ${perReplica}`;
      } else {
        const repP = Math.floor(gpusP / perReplica);
        const repD = Math.floor(gpusD / perReplica);
        const prefillReqS = (repP * effectiveB) / Math.max(prefill.T, 1e-9);
        const decodeTokS = (repD * effectiveB) / Math.max(tpot, 1e-9);
        clusterToks = Math.min(prefillReqS * req.output, decodeTokS);
        servingLabel = `分离 ${hw.pdP}:${hw.pdD}`;
      }
    } else {
      const reqS = (replicasTotal * effectiveB) / Math.max(prefill.T + req.output * tpot, 1e-9);
      clusterToks = reqS * req.output;
      servingLabel = "聚合";
    }

    const tpsPerGpu = clusterToks / Math.max(gpus, 1);
    // billable 口径：输入 token（cache hit 段照样计费）+ 输出 token
    const reqPerS = clusterToks / Math.max(req.output, 1);
    const totalTps = reqPerS * (req.input + req.output);
    const tpm = totalTps * 60;
    const tpmOutput = clusterToks * 60;
    // 按每请求的 GPU 时间份额判断哪一段是瓶颈：TTFT vs 全部输出 token 的 decode 时间
    const prefillHeavy = prefill.T >= req.output * tpot;
    const bound = prefillHeavy ? prefill : decode;
    return {
      prefill,
      decode,
      ttft: prefill.T,
      tpot,
      tpsPerGpu,
      tpm,
      tpmOutput,
      totalTps,
      reqPerS,
      bottleneck: bound.bound,
      boundPhase: prefillHeavy ? "Prefill" : "Decode",
      effectiveB,
      kvMaxB,
      kvLimited,
      servingLabel,
      weights,
      usable,
      fits,
      perReplica,
      plan: commPlan(hw),
      measuredRatio: model.measuredTps ? model.measuredTps / Math.max(tpsPerGpu, 1e-9) : null,
      prefillCost,
      decodeCost,
    };
  }

  function divisors(n) {
    const cap = Math.max(1, n | 0);
    const xs = [];
    for (let i = 1; i <= cap; i += 1) if (cap % i === 0) xs.push(i);
    return xs;
  }

  function largestDivisorAtMost(n, cap) {
    const num = Math.max(1, n | 0);
    const limit = Math.min(Math.max(1, cap | 0), num);
    for (let i = limit; i >= 1; i -= 1) if (num % i === 0) return i;
    return 1;
  }

  function deriveServing(model, cluster) {
    const gpn = Math.max(1, cluster.gpn || 8);
    const d = model.dims;
    const cap = cluster.hbmCapBytes * 0.88;
    const preferTpAttn = (d.kdaLayers || 0) > (d.mlaLayers || 0);
    const attnMode = preferTpAttn ? "tp" : "dp";
    const layout = "overlap";
    const pack = (world, pp) => ({
      tp: largestDivisorAtMost(d.heads || 1, world),
      ep: (d.experts || 1) > 1 ? largestDivisorAtMost(d.experts, world) : 1,
      pp,
      layout,
      attnMode,
    });
    // 8 卡节点从节点宽度起搜，保持 H200 对照；NVL72 这类大 NVLink 域
    // 从 1 卡起搜最小能装下的 replica，再往整机里叠份数（算总体 TPM）。
    const rackScale = gpn >= 32;
    let world = rackScale ? 1 : gpn;
    let pp = 1;
    let cur = pack(world, pp);
    const weightsOf = (s) => weightBytesPerGpu(model, s.tp, s.ep, s.pp);
    while (weightsOf(cur) > cap && world < 256) {
      world *= 2;
      cur = pack(world, pp);
    }
    while (weightsOf(cur) > cap && pp * 2 <= (d.layers || 1)) {
      pp *= 2;
      cur = pack(world, pp);
    }
    const replica = replicaGpus(cur);
    const nodes = Math.max(1, Math.ceil(replica / gpn));
    const onNode = Math.max(cur.tp, cur.ep) <= gpn;
    const clusterGpus = Math.max(1, (cluster.nodes || 1) * gpn);
    const packs = Math.max(1, Math.floor(clusterGpus / replica));
    const packNote = rackScale && onNode && packs > 1
      ? ` · 整机 ${clusterGpus} 卡可放 ${packs} 份`
      : "";
    const why = preferTpAttn
      ? (onNode
        ? `KDA ${d.kdaLayers}/${d.layers} 层 → Attention 走 TP · replica ${replica} 卡 / ${nodes} 节点 · TP=${cur.tp}∩EP=${cur.ep}${packNote}`
        : `KDA 为主，单节点 ${gpn} 卡装不下权重 · replica 占 ${nodes} 个节点（TP${cur.tp}∩EP${cur.ep}）`)
      : (onNode
        ? (cur.pp === 1
          ? `单节点 ${gpn} 卡 · TP${cur.tp} 切 dense/MLA · EP${cur.ep} DeepEP · Attention DP 切 KV${packNote}`
          : `单节点仍装不下全部权重，PP=${cur.pp} · replica 占 ${nodes} 个节点`)
        : `单节点 ${gpn} 卡装不下权重，replica 占 ${nodes} 个节点 · EP=${cur.ep}`);
    return { ...cur, nodes, servingMode: "aggregated", pdP: 1, pdD: 1, why };
  }

  function suggestPlacement(model, cluster, req) {
    const serving = deriveServing(model, cluster);
    const hw = { ...cluster, nodes: serving.nodes, ...serving };
    const est = estimate(model, hw, req);
    return { serving, est, plan: est.plan, why: serving.why };
  }

  window.BallparkRoofline = {
    zero,
    add,
    nodeCost,
    timeOf,
    estimate,
    makeCtx,
    replicaGpus,
    commPlan,
    deriveServing,
    suggestPlacement,
    kvBytesPerSequence,
    weightBytesPerGpu,
    flopScale,
  };
}());
