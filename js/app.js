(function () {
  const $ = (sel) => document.querySelector(sel.startsWith("#") ? sel : `#${sel}`);
  const models = window.BALLPARK_MODELS;
  let selectedId = models[0].id;
  let mode = "prefill";
  let branch = "kda";
  let activeNodeId = "hidden_in";
  let visited = new Set();
  let playing = false;
  let playSteps = [];
  let playIndex = -1;
  let playTimer = null;
  let layerClock = 0;
  let zoom = "fit";
  let currentScale = 1;
  let explainId = "overview";

  const fmt = (n, d = 1) => n.toLocaleString("en-US", { maximumFractionDigits: d });
  const compact = (n) => {
    const abs = Math.abs(n);
    if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
    if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return fmt(n, abs >= 1 ? 1 : 2);
  };
  const fmtTime = (s) => {
    if (!Number.isFinite(s) || s <= 0) return "—";
    if (s >= 1) return `${s.toFixed(2)} s`;
    if (s >= 1e-3) return `${(s * 1e3).toFixed(1)} ms`;
    return `${(s * 1e6).toFixed(0)} µs`;
  };
  const fmtBytes = (n) => {
    if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TB`;
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
    return `${fmt(n, 0)} B`;
  };
  const fmtFlops = (n) => {
    if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PFLOPs`;
    if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TFLOPs`;
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GFLOPs`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MFLOPs`;
    return compact(n);
  };

  function selectedModel() {
    return models.find((item) => item.id === selectedId) || models[0];
  }

  const CACHE_LABEL = {
    "mla-compressed": "compressed KV",
    "hybrid-kda-mla": "KDA state + MLA KV",
    "mla-absorbed": "absorbed KV",
    "paged-gqa": "paged GQA KV",
  };

  function servingFor(model) {
    const item = typeof model === "string" ? models.find((m) => m.id === model) : model;
    const derived = window.BallparkRoofline.deriveServing(item, readCluster());
    return {
      ...derived,
      mtpAccept: item.dims.mtpAcceptDefault,
      measuredTps: item.measuredTps ?? "",
    };
  }

  function ensureMinNodes(serving) {
    const gpn = Math.max(1, +$("#gpn").value || 8);
    const replica = window.BallparkRoofline.replicaGpus(serving);
    const need = serving.nodes || Math.max(1, Math.ceil(replica / gpn));
    if (+$("#nodes").value < need) $("#nodes").value = need;
  }

  function readCluster() {
    const mfu = Math.max(0.05, +$("#efficiency").value / 100);
    const hbmEff = Math.max(0.05, +$("#hbmeff").value / 100);
    const neteff = Math.max(0.05, +$("#neteff").value / 100);
    return {
      nodes: Math.max(1, +$("#nodes").value || 1),
      gpn: Math.max(1, +$("#gpn").value || 1),
      flopsPerGpu: Math.max(1, +$("#compute").value) * 1e12 * mfu,
      hbmPerGpu: Math.max(0.1, +$("#hbm").value) * 1e12 * hbmEff,
      hbmCapBytes: Math.max(8, +$("#hbmCap").value) * 1e9,
      netPerGpu: Math.max(1, +$("#network").value) * 1e9 * neteff,
      intraPerGpu: Math.max(0.1, +$("#nvlink").value) * 1e12 * neteff,
    };
  }

  function mergeHw(cluster, serving) {
    return {
      ...cluster,
      tp: serving.tp,
      ep: serving.ep,
      pp: serving.pp,
      servingMode: serving.servingMode,
      pdP: serving.pdP,
      pdD: serving.pdD,
      layout: serving.layout || "overlap",
      attnMode: serving.attnMode || "dp",
    };
  }

  function readState() {
    const serving = servingFor(selectedModel());
    return {
      hw: mergeHw(readCluster(), serving),
      req: {
        input: +$("#input").value * 1000,
        output: +$("#output").value,
        cache: +$("#cache").value / 100,
        batch: Math.max(1, +$("#batch").value || 1),
        mtpAccept: serving.mtpAccept,
      },
      measured: +serving.measuredTps || 0,
    };
  }

  function topologyCheck(model, serving, gpus) {
    const d = model.dims;
    const replica = window.BallparkRoofline.replicaGpus({
      tp: serving.tp, ep: serving.ep, pp: serving.pp, layout: serving.layout,
    });
    const flags = { tp: false, ep: false, pp: false };
    const issues = [];
    if (d.heads && d.heads % serving.tp !== 0) {
      flags.tp = true;
      issues.push(`TP 应整除 ${d.heads} heads`);
    }
    if (d.experts && d.experts % serving.ep !== 0) {
      flags.ep = true;
      issues.push(`EP 应整除 ${d.experts} experts`);
    }
    if (serving.pp > d.layers) {
      flags.pp = true;
      issues.push(`PP 不能超过 ${d.layers} layers`);
    }
    if (replica > gpus) issues.push(`replica ${replica} > ${gpus} GPU`);
    else if (gpus % replica !== 0) {
      issues.push(`${gpus} GPU 不能被 replica ${replica} 整除，会丢掉 ${gpus % replica} 张卡`);
    }
    return { replica, flags, issues };
  }

  function renderTopo(model, serving, gpus) {
    const d = model.dims;
    const facts = [
      model.scale,
      `${d.layers}L`,
      `${d.heads} heads`,
      d.kvHeads ? `${d.kvHeads} KV heads` : null,
      `${d.experts} exp / Top-${d.activeExperts}`,
      CACHE_LABEL[model.cacheKind] || model.cacheKind,
    ].filter(Boolean);
    $("#topoFacts").innerHTML = facts.map((fact) => `<span class="fact">${fact}</span>`).join("");
    const check = topologyCheck(model, serving, gpus);
    const replicaNodes = serving.nodes || Math.max(1, Math.ceil(check.replica / Math.max(1, +$("#gpn").value || 8)));
    const packs = Math.max(0, Math.floor(gpus / check.replica));
    const attn = serving.attnMode === "tp" ? "TP" : "DP";
    const layout = serving.layout === "mesh" ? "网格 TP×EP" : "重叠 TP∩EP";
    const rows = [
      ["TP", serving.tp],
      ["EP", serving.ep],
      ["PP", serving.pp],
      ["Attention", attn],
      ["布局", layout],
      ["Serving", serving.servingMode === "disaggregated" ? `分离 ${serving.pdP}:${serving.pdD}` : "聚合"],
      ["Replica", `${check.replica} 卡 / ${replicaNodes} 节点`],
    ];
    $("#deployDl").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    const note = $("#topoNote");
    if (check.issues.length) {
      note.textContent = check.issues.join(" · ");
      note.classList.add("is-off");
    } else {
      note.textContent = packs > 1
        ? `${model.name} · 当前机群可放 ${packs} 份 replica`
        : `${model.name} · 由架构和卡推导，无需手调`;
      note.classList.remove("is-off");
    }
    $("#commNote").textContent = serving.why || window.BallparkRoofline.commPlan({
      ...serving, gpn: Math.max(1, +$("#gpn").value || 1),
    }).note;
  }

  function syncModeButtons() {
    document.querySelectorAll(".mode[data-mode]").forEach((el) => {
      el.classList.toggle("active", el.dataset.mode === mode);
    });
    document.querySelectorAll(".mode[data-branch]").forEach((el) => {
      el.classList.toggle("active", el.dataset.branch === branch);
    });
  }

  function playSpeed() {
    return Math.max(0.4, (+$("#playSpeed").value || 10) / 10);
  }

  function visibleGraphNodes(model, hw, branchValue) {
    const graph = window.MODEL_TENSOR_GRAPHS[model.id];
    return window.BallparkWalkthrough.topologicalOrder(
      window.BallparkWalkthrough.visibleNodes(graph, {
        ep: hw.ep,
        branch: model.id === "kimi-k3" ? branchValue : null,
      }),
    );
  }

  function buildPlayScript() {
    const model = selectedModel();
    const { hw } = readState();
    const chapters = model.id === "kimi-k3"
      ? [
        { mode: "prefill", branch: "kda", title: "Prefill · KDA 层", kicker: "PREFILL · KDA" },
        { mode: "prefill", branch: "mla", title: "Prefill · Gated MLA 层", kicker: "PREFILL · MLA" },
        { mode: "decode", branch: "kda", title: "Decode · KDA 层", kicker: "DECODE · KDA" },
        { mode: "decode", branch: "mla", title: "Decode · Gated MLA 层", kicker: "DECODE · MLA" },
      ]
      : [
        { mode: "prefill", branch: branch, title: "Prefill · miss tokens · [B, T, H]", kicker: "PREFILL" },
        { mode: "decode", branch: branch, title: "Decode · token 2+ · [B, 1, H] + cache", kicker: "DECODE" },
      ];
    const steps = [];
    chapters.forEach((chapter) => {
      const nodes = visibleGraphNodes(model, hw, chapter.branch);
      nodes.forEach((node, index) => {
        steps.push({
          ...chapter,
          nodeId: node.id,
          hold: index === nodes.length - 1 ? 1.8 : 1,
          indexInChapter: index + 1,
          chapterLen: nodes.length,
        });
      });
    });
    return steps;
  }

  function clearPlayTimer() {
    if (playTimer) {
      clearTimeout(playTimer);
      playTimer = null;
    }
  }

  function stopPlay() {
    playing = false;
    clearPlayTimer();
    updatePlayUi();
    $("#playBtn").classList.remove("is-playing");
  }

  function resetPlay() {
    stopPlay();
    playIndex = -1;
    playSteps = [];
    visited = new Set();
    layerClock = 0;
    mode = "prefill";
    activeNodeId = "hidden_in";
    explainId = "overview";
    zoom = "fit";
    syncModeButtons();
    render();
  }

  function applyStep(step) {
    const chapterChanged = step.mode !== mode || (step.branch && step.branch !== branch);
    if (chapterChanged) {
      visited = new Set();
      layerClock = 0;
    }
    mode = step.mode;
    if (step.branch) branch = step.branch;
    activeNodeId = step.nodeId;
    explainId = `node:${step.nodeId}`;
    visited.add(step.nodeId);
    syncModeButtons();
    render();
  }

  function advancePlay() {
    if (playIndex >= playSteps.length - 1) {
      playing = false;
      clearPlayTimer();
      updatePlayUi("路径结束");
      render();
      return;
    }
    playIndex += 1;
    applyStep(playSteps[playIndex]);
    if (!playing) return;
    const delay = (520 * (playSteps[playIndex].hold || 1)) / playSpeed();
    playTimer = setTimeout(advancePlay, delay);
  }

  function startPlay() {
    clearPlayTimer();
    playSteps = buildPlayScript();
    if (!playSteps.length) return;
    playing = true;
    updatePlayUi();
    if (playIndex < 0 || playIndex >= playSteps.length - 1) {
      playIndex = -1;
      visited = new Set();
      layerClock = 0;
      advancePlay();
      return;
    }
    const delay = (520 * (playSteps[playIndex].hold || 1)) / playSpeed();
    playTimer = setTimeout(advancePlay, delay);
  }

  function updatePlayUi(doneLabel) {
    const btn = $("#playBtn");
    const mid = playIndex >= 0 && playIndex < playSteps.length - 1;
    btn.classList.toggle("is-playing", playing);
    btn.textContent = playing
      ? "❚❚ 播放中…"
      : mid
        ? "▶ 继续播放"
        : "▶ 播放路径";
    const total = playSteps.length || 1;
    const pct = playIndex < 0 ? 0 : ((playIndex + 1) / total) * 100;
    $("#playBar").style.width = `${pct}%`;
    const step = playSteps[playIndex];
    if (doneLabel) {
      $("#playStatus").textContent = `${doneLabel} · ${playSteps.length} 步`;
    } else if (step) {
      $("#playStatus").textContent = `${step.kicker} ${step.indexInChapter}/${step.chapterLen} · ${playIndex + 1}/${playSteps.length}`;
    } else {
      $("#playStatus").textContent = "Prefill → Decode";
    }
    $("#pathClock").textContent = `层内 ${fmtTime(layerClock)}`;
    $("#speedOut").textContent = `${playSpeed().toFixed(1)}×`;
  }

  function renderTabs() {
    $("#modelTabs").innerHTML = models.map((model) => (
      `<button type="button" class="tab${model.id === selectedId ? " active" : ""}" data-model="${model.id}">${model.name}</button>`
    )).join("");
    $("#modelTabs").querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        selectedId = button.dataset.model;
        branch = "kda";
        resetPlay();
      });
    });
  }

  function bindModes() {
    document.querySelectorAll(".mode[data-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        stopPlay();
        mode = button.dataset.mode;
        visited = new Set();
        layerClock = 0;
        if (explainId === "prefill" || explainId === "decode" || explainId === "overview") {
          explainId = mode;
        }
        syncModeButtons();
        render();
      });
    });
    document.querySelectorAll(".mode[data-branch]").forEach((button) => {
      button.addEventListener("click", () => {
        stopPlay();
        branch = button.dataset.branch;
        visited = new Set();
        layerClock = 0;
        syncModeButtons();
        render();
      });
    });
  }

  function bindPlay() {
    $("#playBtn").addEventListener("click", () => {
      if (playing) stopPlay();
      else startPlay();
    });
    $("#pauseBtn").addEventListener("click", stopPlay);
    $("#stepBtn").addEventListener("click", () => {
      stopPlay();
      if (!playSteps.length) playSteps = buildPlayScript();
      if (playIndex >= playSteps.length - 1) {
        playIndex = -1;
        visited = new Set();
        layerClock = 0;
      }
      playIndex += 1;
      applyStep(playSteps[playIndex]);
    });
    $("#resetBtn").addEventListener("click", resetPlay);
    $("#zoomFit").addEventListener("click", () => {
      zoom = "fit";
      render();
    });
    $("#zoomIn").addEventListener("click", () => {
      zoom = Math.min(4, (typeof zoom === "number" ? zoom : currentScale) * 1.25);
      render();
    });
    $("#zoomOut").addEventListener("click", () => {
      zoom = Math.max(0.05, (typeof zoom === "number" ? zoom : currentScale) / 1.25);
      render();
    });
    $("#playSpeed").addEventListener("input", () => {
      $("#speedOut").textContent = `${playSpeed().toFixed(1)}×`;
    });
    $("#explainPrev").addEventListener("click", () => moveExplain(-1));
    $("#explainNext").addEventListener("click", () => moveExplain(1));
    window.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, textarea, button")) return;
      if (event.key === "ArrowLeft") moveExplain(-1);
      if (event.key === "ArrowRight") moveExplain(1);
    });
  }

  function nodeTimes(nodes, model, hw, ctx) {
    const times = new Map();
    let hotId = null;
    let hotT = -1;
    nodes.forEach((node) => {
      const cost = window.BallparkRoofline.nodeCost(node, ctx);
      const timed = window.BallparkRoofline.timeOf(cost, hw);
      times.set(node.id, { cost, timed });
      if (node.costId !== "none" && timed.T > hotT) {
        hotT = timed.T;
        hotId = node.id;
      }
    });
    return { times, hotId };
  }

  function fillInspector(node, ctx, timedEntry) {
    if (!node) return;
    $("#nodeShape").textContent = window.BallparkWalkthrough.resolveShape(node.shape, ctx);
    $("#nodeOp").textContent = node.op;
    if (timedEntry) {
      const { cost, timed } = timedEntry;
      $("#nodeFlops").textContent = fmtFlops(cost.flops);
      $("#nodeRead").textContent = fmtBytes(cost.bytesRead);
      $("#nodeWrite").textContent = fmtBytes(cost.bytesWrite);
      $("#nodeComm").textContent = fmtBytes((cost.comm || 0) + (cost.commIntra || 0));
      $("#nodeTime").textContent = fmtTime(timed.T);
      $("#nodeBound").textContent = timed.T > 0 ? timed.bound : "—";
    } else {
      $("#nodeFlops").textContent = "—";
      $("#nodeRead").textContent = "—";
      $("#nodeWrite").textContent = "—";
      $("#nodeComm").textContent = "—";
      $("#nodeTime").textContent = "—";
      $("#nodeBound").textContent = "—";
    }
  }

  function layerNoteOf(model) {
    const d = model.dims;
    if (d.kdaLayers) return `${d.kdaLayers} KDA + ${d.mlaLayers} MLA`;
    if (d.indexShareGroup) {
      return `${d.denseLayers} dense · IndexShare /${d.indexShareGroup} · ${d.moeLayers} MoE`;
    }
    if (d.denseLayers) return `${d.denseLayers} dense + ${d.moeLayers} MoE`;
    return `× ${d.layers} 层`;
  }

  function currentToc(model, nodes, hw) {
    return window.BallparkExplain.buildToc(model, nodes, { mode, branch, ep: hw.ep });
  }

  function applyExplainItem(item) {
    stopPlay();
    explainId = item.id;
    if (item.id === "prefill" || item.id === "decode") {
      mode = item.id;
      visited = new Set();
      layerClock = 0;
      syncModeButtons();
    }
    if (item.nodeIds && item.nodeIds.length) {
      const model = selectedModel();
      const { hw } = readState();
      const visible = new Set(visibleGraphNodes(model, hw, branch).map((node) => node.id));
      const first = item.nodeIds.find((id) => visible.has(id));
      if (first) {
        activeNodeId = first;
        visited.add(first);
      }
    }
    render();
  }

  function moveExplain(delta) {
    const model = selectedModel();
    const { hw } = readState();
    const nodes = visibleGraphNodes(model, hw, branch);
    const items = currentToc(model, nodes, hw);
    const key = explainId.startsWith("node:")
      ? window.BallparkExplain.chapterIdFor(model, explainId, activeNodeId, nodes, { mode, branch, ep: hw.ep })
      : explainId;
    const index = items.findIndex((item) => item.id === key);
    const next = items[index + delta];
    if (next) applyExplainItem(next);
  }

  function ensureExplainVisible(model, nodes, hw) {
    const items = currentToc(model, nodes, hw);
    if (items.some((item) => item.id === explainId)) return;
    if (explainId.startsWith("node:")) {
      const id = explainId.slice(5);
      if (nodes.some((node) => node.id === id)) return;
    }
    explainId = "overview";
  }

  function renderToc(model, nodes, hw) {
    const items = currentToc(model, nodes, hw);
    const chapterId = window.BallparkExplain.chapterIdFor(
      model, explainId, activeNodeId, nodes, { mode, branch, ep: hw.ep },
    );
    const X = window.BallparkExplain.escapeHtml;
    const titles = window.BallparkExplain.SECTION_TITLE;
    let last = null;
    const parts = ['<div class="toc-kicker">目录</div>'];
    items.forEach((item) => {
      if (item.section !== last) {
        parts.push(`<div class="toc-sec">${X(titles[item.section] || item.section)}</div>`);
        last = item.section;
      }
      const on = item.id === explainId;
      const inn = !on && explainId.startsWith("node:") && item.id === chapterId;
      parts.push(`<button type="button" data-explain="${X(item.id)}" class="${on ? "is-on" : inn ? "is-in" : ""}">${X(item.title)}</button>`);
    });
    $("#walkToc").innerHTML = parts.join("");
    $("#walkToc").querySelectorAll("[data-explain]").forEach((button) => {
      button.addEventListener("click", () => {
        const item = items.find((entry) => entry.id === button.dataset.explain);
        if (item) applyExplainItem(item);
      });
    });
    const active = $("#walkToc").querySelector(".is-on");
    const toc = $("#walkToc");
    if (active && toc) {
      const top = active.offsetTop - toc.clientHeight / 2 + active.clientHeight / 2;
      toc.scrollTop = Math.max(0, top);
    }
  }

  function fillExplain(model, node, nodes, ctx, hw, timedEntry) {
    const page = window.BallparkExplain.page(model, {
      explainId,
      node,
      nodes,
      ctx,
      mode,
      branch,
      ep: hw.ep,
    });
    $("#explainKicker").textContent = page.kicker;
    $("#explainTitle").textContent = page.title;
    $("#explainWhat").textContent = page.what;
    $("#explainMap").textContent = page.map;
    $("#explainPhase").textContent = page.phase;
    $("#explainCost").textContent = page.cost;
    fillInspector(node, ctx, timedEntry);
    $("#explainMetrics").hidden = ["read", "overview", "prefill", "decode", "layers"].includes(explainId);
  }

  function renderMetrics(est, model, state) {
    $("#ttftOut").textContent = fmtTime(est.ttft);
    $("#tpotOut").textContent = fmtTime(est.tpot);
    $("#tpsOut").textContent = compact(est.tpsPerGpu);
    $("#tpmOut").textContent = compact(est.tpm);
    $("#tpmNote").textContent = est.fits
      ? `billable 输入+输出 · output ${compact(est.tpmOutput)}/min · ${est.reqPerS.toFixed(1)} req/s`
      : "billable 输入+输出 / min";
    $("#bottleneckOut").textContent = `${est.boundPhase} · ${est.bottleneck}`;
    $("#peekTtft").textContent = fmtTime(est.ttft);
    $("#peekTpot").textContent = fmtTime(est.tpot);
    $("#peekTps").textContent = est.fits ? compact(est.tpsPerGpu) : "—";
    $("#peekTpm").textContent = est.fits ? compact(est.tpm) : "—";
    $("#peekBound").textContent = `${est.boundPhase} · ${est.bottleneck}`;
    $("#boundLayer").textContent = !est.fits
      ? `replica ${est.perReplica} 装不进机群`
      : est.kvLimited ? "KV 显存限制了 batch" : "当前路径最慢资源";
    $("#kvOut").textContent = `${est.effectiveB} / ${state.req.batch}`;
    $("#kvNote").textContent = est.kvLimited
      ? `显存最多 ${est.kvMaxB} 并发 · 权重 ${fmtBytes(est.weights)}`
      : `KV 上限 ${est.kvMaxB} · 权重 ${fmtBytes(est.weights)} · ${est.servingLabel}`;
    if (!est.fits) {
      $("#tpsOut").textContent = "—";
      $("#tpmOut").textContent = "—";
    }
    const measured = state.measured || model.measuredTps;
    if (measured && Number.isFinite(est.tpsPerGpu) && est.tpsPerGpu > 0) {
      const ratio = measured / est.tpsPerGpu;
      $("#measuredNote").textContent = `对照 ${measured.toFixed(2)} tok/s/GPU${model.measuredNote ? ` · ${model.measuredNote}` : ""} · 实测/模型 = ${ratio.toFixed(2)}×`;
    } else {
      $("#measuredNote").textContent = model.measuredNote || "结果为 roofline 规划值。";
    }
  }

  function renderCompare(state) {
    const cluster = readCluster();
    $("#compareBody").innerHTML = models.map((model) => {
      const serving = servingFor(model);
      const hw = mergeHw(cluster, serving);
      const req = { ...state.req, mtpAccept: serving.mtpAccept };
      const est = window.BallparkRoofline.estimate(model, hw, req);
      const current = model.id === selectedId;
      return `<button type="button" class="cmp${current ? " current" : ""}" data-model="${model.id}">
        <strong>${model.name}</strong>
        <div class="cmp-row">
          <span>TTFT ${fmtTime(est.ttft)}</span>
          <span>TPOT ${fmtTime(est.tpot)}</span>
          <b>${est.fits ? `${compact(est.tpm)} TPM` : "装不下"}</b>
        </div>
      </button>`;
    }).join("");
    $("#compareBody").querySelectorAll("[data-model]").forEach((el) => {
      el.addEventListener("click", () => {
        if (selectedId === el.dataset.model) return;
        selectedId = el.dataset.model;
        branch = "kda";
        resetPlay();
      });
    });
  }

  function syncShellOffsets() {
    const header = document.querySelector("header.chrome");
    const drawer = document.getElementById("throughput");
    const root = document.documentElement.style;
    root.setProperty("--header-h", `${header ? header.offsetHeight : 0}px`);
    root.setProperty("--drawer-h", `${drawer ? drawer.offsetHeight : 0}px`);
  }

  function drawerIsOpen() {
    return $("#throughput").classList.contains("is-open");
  }

  function setDrawer(open) {
    const el = $("#throughput");
    el.classList.toggle("is-open", open);
    el.setAttribute("aria-expanded", open ? "true" : "false");
    $("#drawerToggleLabel").textContent = open ? "收起" : "展开";
    try { sessionStorage.setItem("ballpark-drawer-open", open ? "1" : "0"); } catch (_err) { /* ignore */ }
    window.requestAnimationFrame(syncShellOffsets);
    window.setTimeout(syncShellOffsets, 320);
  }

  function bindDrawer() {
    $("#drawerToggle").addEventListener("click", () => setDrawer(!drawerIsOpen()));
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && drawerIsOpen()) setDrawer(false);
    });
    let open = false;
    try { open = sessionStorage.getItem("ballpark-drawer-open") === "1"; } catch (_err) { open = false; }
    setDrawer(open);
  }

  function render() {
    syncShellOffsets();
    const model = selectedModel();
    ensureMinNodes(servingFor(model));
    const graph = window.MODEL_TENSOR_GRAPHS[model.id];
    const state = readState();
    const gpus = state.hw.nodes * state.hw.gpn;
    $("#modelTabs").querySelectorAll(".tab").forEach((el) => {
      el.classList.toggle("active", el.dataset.model === selectedId);
    });
    $("#heroGpus").textContent = `${gpus} × GPU`;
    $("#heroShape").textContent = `${$("#input").value}K → ${state.req.output}`;
    $("#heroMode").textContent = mode === "prefill" ? "Prefill · [B, T, H]" : "Decode · [B, 1, H]";
    $("#inputOut").textContent = `${$("#input").value}K`;
    $("#outputOut").textContent = String(state.req.output);
    $("#cacheOut").textContent = `${$("#cache").value}%`;
    $("#batchOut").textContent = String(state.req.batch);
    $("#pipePrefill").classList.toggle("on", mode === "prefill");
    $("#pipeDecode").classList.toggle("on", mode === "decode");
    $("#k3Branch").hidden = model.id !== "kimi-k3";
    $("#chapterKicker").textContent = mode === "prefill"
      ? (model.id === "kimi-k3" ? `PREFILL · ${branch === "mla" ? "MLA" : "KDA"}` : "PREFILL")
      : (model.id === "kimi-k3" ? `DECODE · ${branch === "mla" ? "MLA" : "KDA"}` : "DECODE");
    $("#chapterTitle").textContent = mode === "prefill"
      ? "miss tokens · [B, T, H]"
      : "token 2+ · [B, 1, H] + cache";

    const est = window.BallparkRoofline.estimate(model, state.hw, state.req);
    const ctx = window.BallparkRoofline.makeCtx(model, state.hw, {
      ...state.req,
      batch: est.effectiveB,
    }, mode);
    const nodes = window.BallparkWalkthrough.visibleNodes(graph, {
      ep: state.hw.ep,
      branch: model.id === "kimi-k3" ? branch : null,
    });
    if (!nodes.some((item) => item.id === activeNodeId)) activeNodeId = nodes[0]?.id;
    const { times, hotId } = nodeTimes(nodes, model, state.hw, ctx);
    layerClock = [...visited].reduce((sum, id) => sum + (times.get(id)?.timed.T || 0), 0);

    const drawn = window.BallparkWalkthrough.render({
      graph,
      ep: state.hw.ep,
      branch: model.id === "kimi-k3" ? branch : null,
      mode,
      ctx,
      zoom,
      dims: model.dims,
      modelId: model.id,
      onZoom: (next) => {
        zoom = next;
        render();
      },
      layerNote: layerNoteOf(model),
      activeId: activeNodeId,
      hotId,
      visitedIds: visited,
      playing,
      onSelect: (id) => {
        stopPlay();
        activeNodeId = id;
        explainId = `node:${id}`;
        visited.add(id);
        render();
      },
      onSelectGroup: (groupId) => {
        const section = window.BallparkExplain.GROUP_SECTION[groupId];
        const item = currentToc(model, nodes, state.hw).find((entry) => entry.section === section);
        if (item) applyExplainItem(item);
      },
      onChapter: (id) => {
        const item = currentToc(model, nodes, state.hw).find((entry) => entry.id === id);
        if (item) applyExplainItem(item);
        else {
          stopPlay();
          explainId = id;
          render();
        }
      },
    });

    currentScale = drawn?.scale || 1;
    $("#zoomPct").textContent = `${Math.round(currentScale * 100)}%`;

    ensureExplainVisible(model, nodes, state.hw);
    const active = nodes.find((item) => item.id === activeNodeId) || nodes[0];
    renderToc(model, nodes, state.hw);
    fillExplain(model, active, nodes, ctx, state.hw, active ? times.get(active.id) : null);
    $("#graphCaption").textContent = `${graph.caption} 证据：${graph.evidenceLevel}。`;
    $("#graphSources").innerHTML = (graph.sources || []).map((source) => (
      `<a href="${source.url}" target="_blank" rel="noreferrer">${source.label} ↗</a>`
    )).join("");

    renderMetrics(est, model, state);
    renderCompare(state);
    renderTopo(model, servingFor(model), gpus);
    updatePlayUi(playing || playIndex < playSteps.length - 1 ? null : (playIndex >= 0 ? "路径结束" : null));
    syncShellOffsets();
  }

  try {
    renderTabs();
    bindModes();
    bindDrawer();
    bindPlay();
    document.querySelectorAll(".jigs input, .jigs select").forEach((el) => {
      if (el.id === "playSpeed") return;
      el.addEventListener("input", () => render());
    });
    window.addEventListener("resize", () => render());
    render();
  } catch (err) {
    console.error(err);
  }
}());
