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

  function readState() {
    const eta = Math.max(0.05, +$("#efficiency").value / 100);
    const neteff = Math.max(0.05, +$("#neteff").value / 100);
    return {
      hw: {
        nodes: Math.max(1, +$("#nodes").value || 1),
        gpn: Math.max(1, +$("#gpn").value || 1),
        flopsPerGpu: Math.max(1, +$("#compute").value) * 1e12 * eta,
        hbmPerGpu: Math.max(0.1, +$("#hbm").value) * 1e12 * eta,
        hbmCapBytes: Math.max(8, +$("#hbmCap").value) * 1e9,
        netPerGpu: Math.max(1, +$("#network").value) * 1e9 * neteff * eta,
        intraPerGpu: 450e9 * eta,
        tp: Math.max(1, +$("#tp").value || 1),
        ep: Math.max(1, +$("#ep").value || 1),
        pp: Math.max(1, +$("#pp").value || 1),
        servingMode: $("#servingMode").value,
        pdP: Math.max(0, +$("#pdP").value || 0),
        pdD: Math.max(0, +$("#pdD").value || 0),
      },
      req: {
        input: +$("#input").value * 1000,
        output: +$("#output").value,
        cache: +$("#cache").value / 100,
        batch: Math.max(1, +$("#batch").value || 1),
        mtpAccept: Math.max(0.25, +$("#mtpAccept").value || 1),
      },
      measured: +$("#measuredTps").value || 0,
    };
  }

  function applyModelDefaults(model, { topology } = {}) {
    if (topology) {
      const p = model.parallel;
      $("#tp").value = p.tp;
      $("#ep").value = p.ep;
      $("#pp").value = p.pp;
      $("#pdP").value = p.pdP;
      $("#pdD").value = p.pdD;
      $("#servingMode").value = p.servingMode;
    }
    $("#mtpAccept").value = model.dims.mtpAcceptDefault;
    $("#measuredTps").value = model.measuredTps ?? "";
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
        : "▶ 播放完整推理路径";
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
        applyModelDefaults(selectedModel());
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
      zoom = Math.min(1.6, (typeof zoom === "number" ? zoom : currentScale) + 0.15);
      render();
    });
    $("#zoomOut").addEventListener("click", () => {
      zoom = Math.max(0.25, (typeof zoom === "number" ? zoom : currentScale) - 0.15);
      render();
    });
    $("#playSpeed").addEventListener("input", () => {
      $("#speedOut").textContent = `${playSpeed().toFixed(1)}×`;
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
    $("#nodeKind").textContent = `${node.kind.toUpperCase()} · ${mode.toUpperCase()}`;
    $("#nodeTitle").textContent = node.label;
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
    $("#nodeDetail").textContent = node.detail;
  }

  function renderMetrics(est, model, state) {
    $("#ttftOut").textContent = fmtTime(est.ttft);
    $("#tpotOut").textContent = fmtTime(est.tpot);
    $("#tpsOut").textContent = compact(est.tpsPerGpu);
    $("#tpmOut").textContent = compact(est.tpm);
    $("#bottleneckOut").textContent = `${est.boundPhase} · ${est.bottleneck}`;
    $("#boundLayer").textContent = est.kvLimited ? "KV 显存限制了 batch" : "当前路径最慢资源";
    $("#kvOut").textContent = `${est.effectiveB} / ${state.req.batch}`;
    $("#kvNote").textContent = est.kvLimited
      ? `显存最多 ${est.kvMaxB} 并发 · 权重 ${fmtBytes(est.weights)}`
      : `KV 上限 ${est.kvMaxB} · 权重 ${fmtBytes(est.weights)}`;
    const measured = state.measured || model.measuredTps;
    if (measured && Number.isFinite(est.tpsPerGpu) && est.tpsPerGpu > 0) {
      const ratio = measured / est.tpsPerGpu;
      $("#measuredNote").textContent = `对照 ${measured.toFixed(2)} tok/s/GPU${model.measuredNote ? ` · ${model.measuredNote}` : ""} · 实测/模型 = ${ratio.toFixed(2)}×`;
    } else {
      $("#measuredNote").textContent = model.measuredNote || "结果为 roofline 规划值。";
    }
  }

  function renderCompare(state) {
    $("#compareBody").innerHTML = models.map((model) => {
      const est = window.BallparkRoofline.estimate(model, state.hw, state.req);
      const current = model.id === selectedId;
      return `<tr>
        <td><strong>${model.name}</strong>${current ? " ·" : ""}<br><span class="badge">${model.family}</span></td>
        <td>${fmtTime(est.ttft)}</td>
        <td>${fmtTime(est.tpot)}</td>
        <td><strong>${compact(est.tpm)}</strong></td>
      </tr>`;
    }).join("");
  }

  function render() {
    const model = selectedModel();
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
      layerNote: model.id === "kimi-k3"
        ? `${model.dims.kdaLayers} KDA + ${model.dims.mlaLayers} MLA`
        : `× ${model.dims.layers} 层`,
      activeId: activeNodeId,
      hotId,
      visitedIds: visited,
      playing,
      onSelect: (id) => {
        stopPlay();
        activeNodeId = id;
        visited.add(id);
        render();
      },
    });

    currentScale = drawn?.scale || 1;
    $("#zoomPct").textContent = `${Math.round(currentScale * 100)}%`;

    const active = nodes.find((item) => item.id === activeNodeId) || nodes[0];
    if (active) fillInspector(active, ctx, times.get(active.id));
    $("#graphCaption").textContent = `${graph.caption} 证据：${graph.evidenceLevel}。`;
    $("#graphSources").innerHTML = (graph.sources || []).map((source) => (
      `<a href="${source.url}" target="_blank" rel="noreferrer">${source.label} ↗</a>`
    )).join("");

    renderMetrics(est, model, state);
    renderCompare(state);
    updatePlayUi(playing || playIndex < playSteps.length - 1 ? null : (playIndex >= 0 ? "路径结束" : null));
  }

  try {
    renderTabs();
    bindModes();
    bindPlay();
    applyModelDefaults(selectedModel(), { topology: true });
    document.querySelectorAll(".jigs input, .jigs select").forEach((el) => {
      if (el.id === "playSpeed") return;
      el.addEventListener("input", render);
    });
    window.addEventListener("resize", () => render());
    render();
  } catch (err) {
    console.error(err);
  }
}());
