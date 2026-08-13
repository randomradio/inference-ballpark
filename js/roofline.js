(function () {
  const zero = () => ({ flops: 0, bytesRead: 0, bytesWrite: 0, comm: 0, commIntra: 0 });
  const pack = (c) => ({
    flops: c.flops || 0,
    bytesRead: c.bytesRead || 0,
    bytesWrite: c.bytesWrite || 0,
    comm: c.comm || 0,
    commIntra: c.commIntra || 0,
  });
  const add = (a, b) => {
    a = pack(a); b = pack(b);
    return {
      flops: a.flops + b.flops,
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
      bytesRead: c.bytesRead * n,
      bytesWrite: c.bytesWrite * n,
      comm: c.comm * n,
      commIntra: c.commIntra * n,
    };
  };

  function gemm(ctx, dIn, dOut) {
    const { B, T, wBytes, aBytes } = ctx;
    return {
      flops: 2 * B * T * dIn * dOut,
      bytesRead: dIn * dOut * wBytes + B * T * dIn * aBytes,
      bytesWrite: B * T * dOut * aBytes,
      comm: 0,
    };
  }

  function rms(ctx, dim) {
    return {
      flops: 4 * ctx.B * ctx.T * dim,
      bytesRead: ctx.B * ctx.T * dim * ctx.aBytes,
      bytesWrite: ctx.B * ctx.T * dim * ctx.aBytes,
      comm: 0,
    };
  }

  function swiglu(ctx, h, hidden, nExperts) {
    return scale(add(gemm(ctx, h, hidden * 2), gemm(ctx, hidden, h)), nExperts);
  }

  function attnQKAV(ctx, heads, tq, tk, dHead) {
    const { B, aBytes } = ctx;
    const flops = 4 * B * heads * tq * tk * dHead;
    const bytesRead = B * heads * (tq * dHead + 2 * tk * dHead) * aBytes;
    const bytesWrite = B * heads * tq * dHead * aBytes;
    return { flops, bytesRead, bytesWrite, comm: 0 };
  }

  function allToAll(ctx, h, topk) {
    const { B, T, aBytes, ep } = ctx;
    const tokens = (B * T * topk) / Math.max(1, ep);
    const comm = 2 * tokens * h * aBytes;
    return { flops: 0, bytesRead: 0, bytesWrite: 0, comm };
  }

  function tpAllReduce(ctx, h) {
    if (ctx.tp <= 1) return zero();
    return { flops: 0, bytesRead: 0, bytesWrite: 0, comm: 0, commIntra: 2 * ctx.B * ctx.T * h * ctx.aBytes };
  }

  function nodeCost(node, ctx) {
    const d = ctx.dims;
    switch (node.costId) {
      case "gemm":
        return gemm(ctx, node.dIn, node.dOut);
      case "embed":
        return {
          flops: 0,
          bytesRead: ctx.B * ctx.T * d.H * ctx.aBytes + d.vocabulary * d.H * ctx.wBytes / Math.max(ctx.T, 1),
          bytesWrite: ctx.B * ctx.T * d.H * ctx.aBytes,
          comm: 0,
        };
      case "rms":
        return rms(ctx, node.dIn || d.H);
      case "rope":
        return { flops: 8 * ctx.B * ctx.T * (d.heads || 64) * (d.qkRope || 64), bytesRead: 0, bytesWrite: 0, comm: 0 };
      case "indexer_q":
        return gemm(ctx, node.dIn || d.qLatent, (d.indexHeads || 32) * (d.indexDim || 128));
      case "indexer_k":
        return gemm(ctx, d.H, d.indexDim || 128);
      case "indexer_scan": {
        const heads = d.indexHeads || 32;
        const dim = d.indexDim || 128;
        return attnQKAV(ctx, heads, ctx.T, ctx.S, dim / 2);
      }
      case "attn_sparse_mla":
        return attnQKAV(ctx, d.heads, ctx.T, Math.min(ctx.S, d.indexTopK || 2048), d.valueHead || 256);
      case "kv_cache_mla": {
        const width = (d.kvLatent || 512) + (d.qkRope || 64);
        const read = ctx.mode === "decode";
        const bytes = ctx.B * (read ? ctx.S : ctx.T) * width * ctx.kvBytes;
        return { flops: 0, bytesRead: read ? bytes : 0, bytesWrite: read ? ctx.B * width * ctx.kvBytes : bytes, comm: 0 };
      }
      case "kv_cache_gqa": {
        const width = (d.kvHeads || 4) * (d.headDim || 128) * 2;
        const read = ctx.mode === "decode";
        const stored = ctx.B * ctx.S * width * ctx.kvBytes;
        const selected = (d.indexTopK + (d.localBlocks || 1)) * (d.blockSize || 128);
        const traffic = ctx.B * Math.min(ctx.S, selected) * width * ctx.kvBytes;
        return { flops: 0, bytesRead: read ? traffic : 0, bytesWrite: read ? ctx.B * width * ctx.kvBytes : ctx.B * ctx.T * width * ctx.kvBytes, comm: 0, stored };
      }
      case "kda_qkv":
        return gemm(ctx, d.H, 3 * d.heads * d.headDim);
      case "kda_conv":
        return { flops: 3 * ctx.B * ctx.T * d.heads * d.headDim * 4, bytesRead: ctx.B * ctx.T * d.heads * d.headDim * 3 * ctx.aBytes, bytesWrite: 0, comm: 0 };
      case "kda_gate":
        return gemm(ctx, d.H, d.heads * d.headDim);
      case "attn_kda": {
        const state = ctx.B * d.heads * d.headDim * d.headDim * ctx.kvBytes;
        return {
          flops: 2 * ctx.B * ctx.T * d.heads * d.headDim * d.headDim,
          bytesRead: state + ctx.B * ctx.T * d.heads * d.headDim * ctx.aBytes,
          bytesWrite: state,
          comm: 0,
        };
      }
      case "mla_q":
        return add(gemm(ctx, d.H, d.qLatent), gemm(ctx, d.qLatent, d.heads * ((d.qkNope || 128) + (d.qkRope || 64))));
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
        return swiglu(ctx, node.dIn || d.H, node.dOut || d.expertHidden, d.activeExperts);
      case "moe_shared":
        return swiglu(ctx, node.dIn || d.H, node.dOut || d.expertHidden, d.sharedExperts || 1);
      case "all_to_all":
        return allToAll(ctx, node.dIn || d.latent || d.H, d.activeExperts);
      default:
        return zero();
    }
  }

  function layerAttention(model, ctx, kind) {
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
        add(gemm(ctx, d.H, d.indexDim), attnQKAV(ctx, d.indexHeads, ctx.T, ctx.S, d.indexDim / 2)),
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
      c = add(c, gemm(ctx, d.latent, d.experts));
      c = add(c, swiglu(ctx, d.latent, d.expertHidden, d.activeExperts));
      c = add(c, swiglu(ctx, d.latent, d.expertHidden, d.sharedExperts));
      c = add(c, gemm(ctx, d.latent, d.H));
    } else if (kind === "moe") {
      c = add(c, gemm(ctx, d.H, d.experts));
      c = add(c, swiglu(ctx, d.H, d.expertHidden, d.activeExperts));
      c = add(c, swiglu(ctx, d.H, d.expertHidden, d.sharedExperts || 1));
    } else if (kind === "dense") {
      const hidden = d.denseHidden || d.expertHidden * 4;
      c = add(c, swiglu(ctx, d.H, hidden, 1));
    }
    if (ctx.ep > 1 && kind !== "dense") c = add(c, allToAll(ctx, h, d.activeExperts));
    c = add(c, tpAllReduce(ctx, d.H));
    return c;
  }

  function stackCost(model, ctx) {
    const d = model.dims;
    let c = zero();
    if (model.id === "glm-52") {
      const full = { ...ctx, indexShareSkip: false };
      const share = { ...ctx, indexShareSkip: true };
      const group = d.indexShareGroup || 4;
      const fullN = Math.ceil(d.layers / group);
      const shareN = d.layers - fullN;
      c = add(c, scale(layerAttention(model, full, "dsa-mla"), fullN));
      c = add(c, scale(layerAttention(model, share, "dsa-mla"), shareN));
      c = add(c, scale(layerFfn(model, ctx, "dense"), d.denseLayers));
      c = add(c, scale(layerFfn(model, ctx, "moe"), d.moeLayers));
    } else if (model.id === "kimi-k3") {
      c = add(c, scale(layerAttention(model, ctx, "kda"), d.kdaLayers));
      c = add(c, scale(layerAttention(model, ctx, "gated-mla"), d.mlaLayers));
      c = add(c, scale(layerFfn(model, ctx, "latent-moe"), d.layers));
    } else if (model.id === "kimi-k25") {
      c = add(c, scale(layerAttention(model, ctx, "mla"), d.layers));
      c = add(c, scale(layerFfn(model, ctx, "dense"), d.denseLayers));
      c = add(c, scale(layerFfn(model, ctx, "moe"), d.moeLayers));
    } else {
      c = add(c, scale(layerAttention(model, ctx, "dense-gqa"), d.denseLayers));
      c = add(c, scale(layerAttention(model, ctx, "sparse-gqa"), d.moeLayers));
      c = add(c, scale(layerFfn(model, ctx, "dense"), d.denseLayers));
      c = add(c, scale(layerFfn(model, ctx, "moe"), d.moeLayers));
    }
    return c;
  }

  function onceCost(model, ctx) {
    const d = model.dims;
    const last = { ...ctx, T: 1 };
    return add(add(nodeCost({ costId: "embed" }, ctx), rms(last, d.H)), gemm(last, d.H, d.vocabulary));
  }

  function timeOf(cost, hw) {
    cost = pack(cost);
    const replica = Math.max(1, hw.tp * hw.ep * hw.pp);
    const Tc = cost.flops / replica / hw.flopsPerGpu;
    const Tm = (cost.bytesRead + cost.bytesWrite) / replica / hw.hbmPerGpu;
    const intra = hw.intraPerGpu || hw.hbmPerGpu * 0.18;
    const TnNet = cost.comm / Math.max(hw.netPerGpu, 1e-9);
    const TnIntra = cost.commIntra / Math.max(intra, 1e-9);
    const Tn = Math.max(TnNet, TnIntra);
    const T = Math.max(Tc, Tm, Tn);
    const bound = Tc >= Tm && Tc >= Tn ? "计算" : Tm >= Tn ? "HBM" : TnNet >= TnIntra ? "网络" : "节点内";
    return { Tc, Tm, Tn, T, bound };
  }

  function weightBytesPerGpu(model, tp, ep) {
    const d = model.dims;
    const w = d.weightBytes;
    const dense = (h, hidden) => 3 * h * hidden * w;
    let layer = 0;
    if (model.id === "glm-52") {
      layer += (d.H * d.qLatent + d.qLatent * d.heads * (d.qkNope + d.qkRope) + d.H * (d.kvLatent + d.qkRope) + d.kvLatent * d.heads * (d.qkNope + d.valueHead) + d.heads * d.valueHead * d.H) * w;
      layer += (d.experts / ep) * dense(d.H, d.expertHidden) + dense(d.H, d.expertHidden);
    } else if (model.id === "kimi-k3") {
      layer += (3 * d.H * d.heads * d.headDim + d.H * (d.kvLatent + d.qkRope) + d.H * d.qLatent) * w;
      layer += (d.H * d.latent + (d.experts / ep) * dense(d.latent, d.expertHidden) + d.sharedExperts * dense(d.latent, d.expertHidden) + d.latent * d.H) * w;
    } else if (model.id === "kimi-k25") {
      layer += (d.H * d.qLatent + d.qLatent * 12288 + d.H * 576 + 512 * 16384 + 8192 * d.H) * w;
      layer += ((d.experts / ep) * dense(d.H, d.expertHidden) + dense(d.H, d.expertHidden)) * 1;
    } else {
      layer += (d.H * d.heads * d.headDim + d.H * d.kvHeads * d.headDim * 2 + d.heads * d.headDim * d.H) * w;
      layer += (d.experts / ep) * dense(d.H, d.expertHidden) + dense(d.H, d.expertHidden);
    }
    const stack = layer * d.layers;
    const heads = (d.vocabulary * d.H * 2) * w;
    return (stack + heads) / Math.max(1, tp);
  }

  function kvBytesPerSequence(model, S, B) {
    const d = model.dims;
    const k = d.kvBytes;
    if (model.id === "glm-52") {
      return B * S * d.layers * ((d.kvLatent + d.qkRope) + (d.indexDim || 128)) * k;
    }
    if (model.id === "kimi-k3") {
      const state = B * d.kdaLayers * d.heads * d.headDim * d.headDim * k;
      const mla = B * S * d.mlaLayers * (d.kvLatent + d.qkRope) * k;
      return state + mla;
    }
    if (model.id === "kimi-k25") {
      return B * S * d.layers * (d.kvLatent + d.qkRope) * k;
    }
    return B * S * d.layers * d.kvHeads * d.headDim * 2 * k;
  }

  function makeCtx(model, hw, req, mode) {
    const Tmiss = Math.max(1, Math.round(req.input * (1 - req.cache)));
    const T = mode === "prefill" ? Tmiss : 1;
    const S = req.input;
    return {
      mode,
      B: req.batch,
      T,
      S,
      dims: model.dims,
      wBytes: model.dims.weightBytes,
      aBytes: model.dims.kvBytes,
      kvBytes: model.dims.kvBytes,
      tp: hw.tp,
      ep: hw.ep,
    };
  }

  function replicaGpus(hw) {
    return Math.max(1, hw.tp * hw.ep * hw.pp);
  }

  function estimate(model, hw, req) {
    const gpus = hw.nodes * hw.gpn;
    const weights = weightBytesPerGpu(model, hw.tp, hw.ep);
    const usable = Math.max(0, hw.hbmCapBytes * 0.88 - weights);
    const perSeq = kvBytesPerSequence(model, req.input, 1);
    const kvMaxB = perSeq > 0 ? Math.max(1, Math.floor(usable / perSeq)) : req.batch;
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

    const perReplica = replicaGpus(hw);
    const replicasTotal = Math.max(1, Math.floor(gpus / perReplica));

    let clusterToks = 0;
    let servingLabel = "聚合";
    if (hw.servingMode === "disaggregated" && hw.pdP + hw.pdD > 0) {
      const share = hw.pdP / (hw.pdP + hw.pdD);
      const gpusP = Math.max(perReplica, Math.floor(gpus * share / perReplica) * perReplica);
      const gpusD = Math.max(perReplica, gpus - gpusP);
      const repP = Math.max(1, Math.floor(gpusP / perReplica));
      const repD = Math.max(1, Math.floor(gpusD / perReplica));
      const prefillReqS = (repP * effectiveB) / Math.max(prefill.T, 1e-9);
      const decodeTokS = (repD * effectiveB) / Math.max(tpot, 1e-9);
      clusterToks = Math.min(prefillReqS * req.output, decodeTokS);
      servingLabel = `分离 ${hw.pdP}:${hw.pdD}`;
    } else {
      const reqS = (replicasTotal * effectiveB) / Math.max(prefill.T + req.output * tpot, 1e-9);
      clusterToks = reqS * req.output;
      servingLabel = "聚合";
    }

    const tpsPerGpu = clusterToks / Math.max(gpus, 1);
    const tpm = clusterToks * 60;
    const bound = prefill.T >= tpot ? prefill : decode;
    return {
      prefill,
      decode,
      ttft: prefill.T,
      tpot,
      tpsPerGpu,
      tpm,
      bottleneck: bound.bound,
      boundPhase: prefill.T > tpot * 8 ? "Prefill" : "Decode",
      effectiveB,
      kvMaxB,
      kvLimited,
      servingLabel,
      weights,
      usable,
      measuredRatio: model.measuredTps ? model.measuredTps / Math.max(tpsPerGpu, 1e-9) : null,
      prefillCost,
      decodeCost,
    };
  }

  window.BallparkRoofline = {
    zero,
    add,
    nodeCost,
    timeOf,
    estimate,
    makeCtx,
    kvBytesPerSequence,
    weightBytesPerGpu,
  };
}());
