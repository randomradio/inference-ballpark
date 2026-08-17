(function () {
  const kindColors = {
    activation: "#111",
    weight: "#444",
    operation: "#666",
    cache: "#888",
    communication: "#222",
    module: "#555",
  };
  const LAYER_KIND_COLORS = {
    kda: "#111", mla: "#666", dense: "#bbb", moe: "#111",
    dsa: "#666", share: "#ddd",
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function fmtNum(n) {
    if (n >= 1000) return `${Math.round(n / 1000)}K`;
    return String(n);
  }

  function resolveShape(shape, ctx) {
    return String(shape)
      .replaceAll("{B}", String(ctx.B))
      .replaceAll("{T}", fmtNum(ctx.T))
      .replaceAll("{S}", fmtNum(ctx.S));
  }

  // 几何标度：宽 ∝ log₂ 列维，高 ∝ log₂ 行/token 维（参数按设计规范写死）
  const pxW = (n) => Math.min(320, 26 + 9.5 * Math.log2(Math.max(1, n)));
  const pxH = (n) => Math.min(240, 20 + 7.5 * Math.log2(Math.max(1, n)));

  function dimValue(seg, ctx) {
    const sub = seg
      .replaceAll("{B}", String(ctx.B))
      .replaceAll("{T}", String(ctx.T))
      .replaceAll("{S}", String(ctx.S));
    const nums = sub.replace(/,/g, "").match(/\d+/g);
    if (!nums) return null;
    return nums.map(Number).reduce((a, b) => a * b, 1);
  }

  // 把 shape 字符串解析成数值维度；段内多个数字相乘（{B}×{T} 是一个逻辑维）
  function shapeDims(shape, ctx) {
    const raw = String(shape);
    const bracket = raw.match(/\[([^\]]+)\]/);
    if (!bracket) {
      const tail = raw.match(/(\d[\d,]*)\s*$/);
      if (!tail) return null;
      const n = Number(tail[1].replace(/,/g, ""));
      if (!Number.isFinite(n) || n <= 0) return null;
      return { rows: { value: 1, dynamic: false }, cols: { value: n, dynamic: false }, stack: 0 };
    }
    const segs = bracket[1].split(/,\s+/).map((seg) => {
      const value = dimValue(seg, ctx);
      if (!value) return null;
      return { value, dynamic: /\{[BTS]\}/.test(seg) };
    }).filter(Boolean);
    if (!segs.length) return null;
    if (segs.length === 1) return { rows: { value: 1, dynamic: false }, cols: segs[0], stack: 0 };
    if (segs.length === 2) return { rows: segs[0], cols: segs[1], stack: 0 };
    if (segs.length === 3) {
      return {
        rows: { value: segs[0].value * segs[1].value, dynamic: segs[0].dynamic || segs[1].dynamic },
        cols: segs[2],
        stack: 0,
      };
    }
    // 4 维有两种约定：[B, heads, T, d]（dims[1] 静态 = 重复轴）与
    // [B, S, kvHeads, d]（dims[1] 是 token 维，重复轴在 dims[2]）
    if (!segs[1].dynamic) {
      return {
        rows: { value: segs[0].value * segs[2].value, dynamic: segs[0].dynamic || segs[2].dynamic },
        cols: segs[3],
        stack: segs[1].value,
      };
    }
    return {
      rows: { value: segs[0].value * segs[1].value, dynamic: true },
      cols: segs[3],
      stack: segs[2].dynamic ? 0 : segs[2].value,
    };
  }

  function nodeGeometry(item, ctx) {
    const isMatrix = (item.kind === "weight" || item.kind === "module") && item.dIn && item.dOut;
    const dims = isMatrix
      ? { rows: { value: item.dIn, dynamic: false }, cols: { value: item.dOut, dynamic: false }, stack: 0 }
      : shapeDims(item.shape, ctx);
    if (!dims) {
      return { chip: true, width: 120, height: 44, rows: null, cols: null, stack: item.stack || 0, matrix: false };
    }
    return {
      chip: false,
      width: Math.max(56, pxW(dims.cols.value)),
      height: pxH(dims.rows.value),
      rows: dims.rows,
      cols: dims.cols,
      stack: item.stack || dims.stack || 0,
      matrix: isMatrix || item.kind === "weight",
    };
  }

  function fmtDim(dim) {
    if (!dim) return "";
    if (dim.dynamic) {
      if (dim.value >= 1e6) return `${(dim.value / 1e6).toFixed(1)}M`;
      if (dim.value >= 1e3) return `${Math.round(dim.value / 1e3)}K`;
      return String(dim.value);
    }
    return dim.value.toLocaleString("en-US");
  }

  function visibleNodes(graph, { ep, branch }) {
    return graph.nodes.filter((item) => {
      if (item.epOnly && ep <= 1) return false;
      if (item.skipIfEp && ep > 1) return false;
      if (item.layerBranch && branch && item.layerBranch !== branch) return false;
      return true;
    });
  }

  function upstreamIds(nodes, nodeId, result = new Set()) {
    const item = nodes.find((candidate) => candidate.id === nodeId);
    if (!item) return result;
    const visible = new Set(nodes.map((n) => n.id));
    item.dependsOn.forEach((dependency) => {
      if (!visible.has(dependency) || result.has(dependency)) return;
      result.add(dependency);
      upstreamIds(nodes, dependency, result);
    });
    return result;
  }

  // Vertical column: embedding, nested decoder (attention + MoE), output head.
  const GROUP_OF_STAGE = { 0: 0, 1: 1, 2: 2, 3: 2, 4: 3 };
  const GROUP_LABELS = ["EMBEDDING", "ATTENTION", "MOE FFN · SERVING", "OUTPUT HEAD"];
  const groupOf = (item) => GROUP_OF_STAGE[item.stageIndex] ?? 2;

  function computeLayout(nodes, ctx) {
    const visibleIds = new Set(nodes.map((item) => item.id));
    const depthMemo = new Map();
    function depth(item, trail = new Set()) {
      if (depthMemo.has(item.id)) return depthMemo.get(item.id);
      if (trail.has(item.id)) return 0;
      const nextTrail = new Set(trail).add(item.id);
      const parents = item.dependsOn
        .filter((id) => visibleIds.has(id))
        .map((id) => nodes.find((candidate) => candidate.id === id))
        .filter(Boolean);
      const value = parents.length ? Math.max(...parents.map((parent) => depth(parent, nextTrail))) + 1 : 0;
      depthMemo.set(item.id, value);
      return value;
    }

    const rows = new Map();
    nodes.forEach((item) => {
      const itemDepth = depth(item);
      if (!rows.has(itemDepth)) rows.set(itemDepth, []);
      rows.get(itemDepth).push(item);
    });

    const geo = new Map(nodes.map((item) => [item.id, nodeGeometry(item, ctx)]));
    const sideMargin = 96;
    const positions = new Map();
    let y = 40;
    let previousGroup = null;

    // 第一遍：以 axis=0 为基准放置，spine（残差流）居中，其余左右交替侧挂
    const orderedDepths = [...rows.keys()].sort((a, b) => a - b);
    orderedDepths.forEach((itemDepth, rowIndex) => {
      const items = rows.get(itemDepth);
      const group = Math.min(...items.map((item) => GROUP_OF_STAGE[item.stageIndex] ?? 2));
      if (previousGroup !== null && group !== previousGroup) y += 58;
      previousGroup = group;
      const rowH = Math.max(...items.map((item) => geo.get(item.id).height)) + 64;
      const spineItems = items.filter((item) => item.spine);
      const others = items.filter((item) => !item.spine);
      const spineW = spineItems.reduce((sum, item) => sum + geo.get(item.id).width, 0)
        + Math.max(0, spineItems.length - 1) * 24;

      let sx = -spineW / 2;
      spineItems.forEach((item) => {
        const g = geo.get(item.id);
        positions.set(item.id, { x: sx, y, width: g.width, height: g.height, group, row: rowIndex, geo: g });
        sx += g.width + 24;
      });

      if (!spineItems.length) {
        const totalW = others.reduce((sum, item) => sum + geo.get(item.id).width, 0)
          + Math.max(0, others.length - 1) * 48;
        let ox = -totalW / 2;
        others.forEach((item) => {
          const g = geo.get(item.id);
          positions.set(item.id, { x: ox, y, width: g.width, height: g.height, group, row: rowIndex, geo: g });
          ox += g.width + 48;
        });
      } else {
        let rx = spineW / 2 + 48;
        let lx = -spineW / 2 - 48;
        others.forEach((item, index) => {
          const g = geo.get(item.id);
          if (index % 2 === 0) {
            positions.set(item.id, { x: rx, y, width: g.width, height: g.height, group, row: rowIndex, geo: g });
            rx += g.width + 48;
          } else {
            positions.set(item.id, { x: lx - g.width, y, width: g.width, height: g.height, group, row: rowIndex, geo: g });
            lx -= g.width + 48;
          }
        });
      }
      y += rowH;
    });

    // 第二遍：按实际跨度定画布宽，把 axis=0 平移到画布中轴
    let minX = Infinity;
    let maxX = -Infinity;
    positions.forEach((box) => {
      minX = Math.min(minX, box.x);
      maxX = Math.max(maxX, box.x + box.width);
    });
    const width = Math.max(560, maxX - minX + sideMargin * 2);
    const shift = sideMargin - minX;
    positions.forEach((box) => { box.x += shift; });
    const axis = shift;
    const height = y + 24;

    return { width, height, positions, visibleIds, axis };
  }

  function groupBoxesOf(positions) {
    const map = new Map();
    positions.forEach((box) => {
      const current = map.get(box.group) || {
        minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity,
      };
      current.minX = Math.min(current.minX, box.x);
      current.minY = Math.min(current.minY, box.y);
      current.maxX = Math.max(current.maxX, box.x + box.width);
      current.maxY = Math.max(current.maxY, box.y + box.height);
      map.set(box.group, current);
    });
    return map;
  }

  function topologicalOrder(nodes) {
    const visible = new Set(nodes.map((item) => item.id));
    const remaining = [...nodes];
    const incoming = new Map(nodes.map((item) => [
      item.id,
      item.dependsOn.filter((id) => visible.has(id)).length,
    ]));
    const ordered = [];
    while (remaining.length) {
      const index = remaining.findIndex((item) => incoming.get(item.id) === 0);
      const next = index >= 0 ? remaining.splice(index, 1)[0] : remaining.shift();
      if (!next) break;
      ordered.push(next);
      remaining.forEach((item) => {
        if (item.dependsOn.includes(next.id)) incoming.set(item.id, incoming.get(item.id) - 1);
      });
    }
    return ordered;
  }

  // 全模型柱：代表层之外的层按真实种类与顺序压成薄片
  function layerSequence(dims) {
    const total = Math.max(1, dims.layers || 1);
    if (dims.kdaLayers) {
      const mla = dims.mlaLayers || 0;
      const seq = [];
      for (let i = 0; i < total; i += 1) {
        const isMla = mla > 0
          && Math.floor(((i + 1) * mla) / total) > Math.floor((i * mla) / total);
        seq.push(isMla ? "mla" : "kda");
      }
      return seq;
    }
    const dense = dims.denseLayers || 0;
    const group = dims.indexShareGroup || 0;
    const seq = [];
    for (let i = 0; i < total; i += 1) {
      if (i < dense) {
        seq.push("dense");
      } else if (group > 0) {
        seq.push((i - dense) % group === 0 ? "dsa" : "share");
      } else {
        seq.push("moe");
      }
    }
    return seq;
  }

  // 中间框展开的那一层：K3 跟当前 KDA/MLA 开关；GLM 是 full DSA；其余是 MoE
  function representedLayerKind(dims, branch) {
    if (dims.kdaLayers) return branch === "mla" ? "mla" : "kda";
    if (dims.indexShareGroup) return "dsa";
    if (dims.moeLayers) return "moe";
    return "dense";
  }

  function remainingLayers(dims, branch) {
    const seq = layerSequence(dims);
    const kind = representedLayerKind(dims, branch);
    const index = seq.indexOf(kind);
    if (index < 0) return seq.slice(1);
    return seq.filter((_, i) => i !== index);
  }

  function layerStack(dims, axis, yTop, branch) {
    const seq = remainingLayers(dims, branch);
    if (!seq.length) return null;
    const h = Math.min(6, Math.max(2, 240 / seq.length));
    const w = pxW(dims.H || 7168);
    const x = axis - w / 2;
    const parts = seq.map((kind, i) => (
      `<rect class="walkthrough-slice" data-stack="1" x="${x}" y="${yTop + i * h}" width="${w}" height="${Math.max(1, h - 0.6)}" fill="${LAYER_KIND_COLORS[kind] || "#111"}" />`
    ));
    const KIND_ORDER = ["dense", "dsa", "share", "kda", "mla", "moe"];
    const counts = seq.reduce((acc, kind) => {
      acc[kind] = (acc[kind] || 0) + 1;
      return acc;
    }, {});
    const breakdown = Object.entries(counts)
      .sort((a, b) => {
        const ra = KIND_ORDER.indexOf(a[0]);
        const rb = KIND_ORDER.indexOf(b[0]);
        return (ra < 0 ? 50 : ra) - (rb < 0 ? 50 : rb);
      })
      .map(([kind, n]) => `${n} ${kind.toUpperCase()}`)
      .join(" + ");
    const height = seq.length * h;
    const note = `<text class="walkthrough-stack-note" data-stack="1" x="${x + w + 16}" y="${yTop + height / 2}">其余 ${seq.length} 层 · ${breakdown}</text>`;
    return { markup: parts.join("") + note, height: height + 48 };
  }

  // 视图状态（模块级，跨 render 保持平移/缩放）
  let view = { scale: 1, tx: 0, ty: 0 };
  let lastModelId = null;
  let lastActiveId = null;
  let pan = null;
  let suppressClickUntil = 0;
  let zoomNotifyTimer = null;
  const current = { onZoom: null };

  function applyView(svg) {
    const g = svg.querySelector(".walkthrough-viewport");
    if (g) g.setAttribute("transform", `translate(${view.tx} ${view.ty}) scale(${view.scale})`);
  }

  function bindStage(svg) {
    if (svg.dataset.bound) return;
    svg.dataset.bound = "1";
    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = svg.getBoundingClientRect();
      const cx = event.clientX - rect.left;
      const cy = event.clientY - rect.top;
      const next = Math.min(4, Math.max(0.05, view.scale * Math.exp(-event.deltaY * 0.0015)));
      view.tx = cx - ((cx - view.tx) * next) / view.scale;
      view.ty = cy - ((cy - view.ty) * next) / view.scale;
      view.scale = next;
      applyView(svg);
      if (zoomNotifyTimer) clearTimeout(zoomNotifyTimer);
      zoomNotifyTimer = setTimeout(() => current.onZoom?.(view.scale), 140);
    }, { passive: false });
    svg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      pan = { x: event.clientX, y: event.clientY, tx: view.tx, ty: view.ty, moved: false };
    });
    window.addEventListener("pointermove", (event) => {
      if (!pan) return;
      const dx = event.clientX - pan.x;
      const dy = event.clientY - pan.y;
      if (!pan.moved && Math.hypot(dx, dy) < 4) return;
      pan.moved = true;
      view.tx = pan.tx + dx;
      view.ty = pan.ty + dy;
      const svgEl = document.getElementById("walkthroughSvg");
      if (svgEl) applyView(svgEl);
      svg.closest(".walkthrough-stage-scroll")?.classList.add("is-panning");
    });
    window.addEventListener("pointerup", () => {
      if (pan?.moved) suppressClickUntil = Date.now() + 160;
      pan = null;
      document.querySelector(".walkthrough-stage-scroll")?.classList.remove("is-panning");
    });
    svg.addEventListener("dblclick", () => current.onZoom?.("fit"));
  }

  function formulaOf(item) {
    if (item.dIn && item.dOut) {
      return { title: "GEMM", lines: ["Y = X W"] };
    }
    if (item.stack && item.stackActive) {
      return { title: "MoE", lines: [`Top-${item.stackActive} / ${item.stack}`] };
    }
    if (item.cacheRole === "readAppend") return { title: "cache", lines: ["read S · append 1"] };
    if (item.cacheRole === "write") return { title: "cache", lines: ["write"] };
    if (/residual/i.test(`${item.id} ${item.op} ${item.label}`)) return { title: "add", lines: ["y = x + f(x)"] };
    if ((item.costId || "").startsWith("attn") || /softmax|attention/i.test(item.costId || item.op || "")) {
      return { title: "Attention", lines: ["QKᵀ → V"] };
    }
    if (item.costId === "rms") return { title: "RMSNorm", lines: ["x / rms(x)"] };
    if (item.costId === "embed") return { title: "lookup", lines: ["id → H"] };
    if (item.costId === "all_to_all") return { title: "all-to-all", lines: [] };
    return { title: item.op || item.kind, lines: [] };
  }

  function flowCardMarkup(item, box, ctx) {
    const f = formulaOf(item);
    const lines = [resolveShape(item.shape, ctx), ...f.lines].filter(Boolean);
    const w = 150;
    const h = 20 + lines.length * 13;
    const x = box.x + box.width + 12;
    const y = box.y;
    return `<g class="walkthrough-flow" pointer-events="none">
      <rect class="walkthrough-flow-card" x="${x}" y="${y}" width="${w}" height="${h}" rx="6" />
      <text class="walkthrough-flow-title" x="${x + 8}" y="${y + 13}">${escapeHtml(f.title)}</text>
      ${lines.map((line, i) => `<text class="walkthrough-flow-line" x="${x + 8}" y="${y + 28 + i * 13}">${escapeHtml(line)}</text>`).join("")}
    </g>`;
  }

  function tokenStrip(box) {
    const n = 5;
    const tw = Math.max(14, box.width / n);
    const cells = [];
    for (let i = 0; i < n; i += 1) {
      const label = i === n - 1 ? "T" : String(i);
      cells.push(`<g class="walkthrough-token">
        <rect class="walkthrough-token-cell" x="${box.x + i * tw}" y="${box.y}" width="${Math.max(12, tw - 2)}" height="${box.height}" rx="3" />
        <text class="walkthrough-token-id" x="${box.x + i * tw + (tw - 2) / 2}" y="${box.y + box.height / 2 + 3}" text-anchor="middle">${label}</text>
      </g>`);
    }
    return cells.join("");
  }

  function tokenMapLines(positions, nodes) {
    const tok = nodes.find((item) => item.id === "token_ids" || item.id === "text_ids");
    const emb = nodes.find((item) => item.id === "embedding");
    if (!tok || !emb) return "";
    const a = positions.get(tok.id);
    const b = positions.get(emb.id);
    if (!a || !b) return "";
    const n = 5;
    const lines = [];
    for (let i = 0; i < n; i += 1) {
      const x0 = a.x + ((i + 0.5) / n) * a.width;
      const x1 = b.x + ((i + 0.5) / n) * b.width;
      lines.push(`<path class="walkthrough-map-line" d="M ${x0} ${a.y + a.height} C ${x0} ${a.y + a.height + 16}, ${x1} ${b.y - 16}, ${x1} ${b.y}" />`);
    }
    return lines.join("");
  }

  function expertGrid(item, box) {
    const total = Math.min(item.stack || 0, 128);
    const active = Math.max(1, Math.min(item.stackActive || 1, total));
    if (!total) return "";
    const cols = 16;
    const rows = Math.ceil(total / cols);
    const pad = 8;
    const cw = (box.width - pad * 2) / cols;
    const ch = (box.height - pad * 2) / rows;
    const step = total / active;
    const lit = new Set();
    for (let k = 0; k < active; k += 1) lit.add(Math.min(total - 1, Math.round(k * step)));
    const cells = [];
    for (let i = 0; i < total; i += 1) {
      const cx = box.x + pad + (i % cols) * cw;
      const cy = box.y + pad + Math.floor(i / cols) * ch;
      cells.push(`<rect x="${cx.toFixed(1)}" y="${cy.toFixed(1)}" width="${Math.max(1, cw - 1).toFixed(1)}" height="${Math.max(1, ch - 1).toFixed(1)}" fill="${kindColors.module}" opacity="${lit.has(i) ? 1 : 0.15}" />`);
    }
    return cells.join("");
  }

  function render(state) {
    const svg = document.getElementById("walkthroughSvg");
    if (!svg || !state?.graph) return;
    const { graph, ep, branch, mode, ctx, activeId, hotId, onSelect, visitedIds, playing } = state;
    const dims = state.dims || {};
    current.onZoom = state.onZoom || null;
    bindStage(svg);

    const nodes = visibleNodes(graph, { ep, branch });
    const activeNode = nodes.find((item) => item.id === activeId) || nodes[0];
    const layout = computeLayout(nodes, ctx);
    const activeUpstream = activeNode ? upstreamIds(nodes, activeNode.id) : new Set();
    const visited = visitedIds instanceof Set ? visitedIds : new Set();

    // 全模型柱：把 OUTPUT HEAD 组下移，在代表层与输出头之间插入压缩层堆
    let stackMarkup = "";
    let blockFrameTop = null;
    let stackBox = null;
    if (dims.layers > 1) {
      const g3 = [...layout.positions.values()].filter((box) => box.group === 3);
      if (g3.length) {
        const g3Top = Math.min(...g3.map((box) => box.y));
        const stack = layerStack(dims, layout.axis, g3Top + 12, branch);
        if (stack) {
          const shift = stack.height;
          const stackW = pxW(dims.H || 7168);
          stackBox = {
            minX: layout.axis - stackW / 2,
            minY: g3Top + 12,
            maxX: layout.axis + stackW / 2 + 220,
            maxY: g3Top + 12 + stack.height,
          };
          layout.positions.forEach((box) => {
            if (box.y >= g3Top) box.y += shift;
          });
          layout.height += shift;
          stackMarkup = stack.markup;
        }
      }
    }
    const groupBoxes = groupBoxesOf(layout.positions);
    const blockBox = groupBoxes.get(1) || groupBoxes.get(2);
    if (blockBox) blockFrameTop = blockBox.minY;

    const scroller = svg.closest(".walkthrough-stage-scroll") || svg.parentElement;
    const cw = Math.max(320, scroller?.clientWidth || 800);
    const ch = Math.max(240, scroller?.clientHeight || 600);

    const fitView = () => {
      view.scale = Math.min(1.2, Math.max(0.05, (cw - 40) / layout.width));
      view.tx = (cw - layout.width * view.scale) / 2;
      view.ty = Math.max(16, Math.min(40, (ch - layout.height * view.scale) / 2));
    };
    const frameRect = (rect, pad = 56) => {
      const w = Math.max(72, rect.maxX - rect.minX);
      const h = Math.max(72, rect.maxY - rect.minY);
      view.scale = Math.min(2.6, Math.max(0.12, Math.min((cw - pad * 2) / w, (ch - pad * 2) / h)));
      view.tx = cw / 2 - ((rect.minX + rect.maxX) / 2) * view.scale;
      view.ty = ch / 2 - ((rect.minY + rect.maxY) / 2) * view.scale;
    };
    const boxRect = (box) => ({
      minX: box.x, minY: box.y, maxX: box.x + box.width, maxY: box.y + box.height,
    });
    if (state.modelId !== lastModelId && !state.panTarget) {
      lastModelId = state.modelId;
      lastActiveId = null;
      fitView();
    } else if (state.zoom === "fit" && !state.panTarget) {
      fitView();
    } else if (!state.panTarget) {
      view.scale = Math.min(4, Math.max(0.05, Number(state.zoom) || 1));
    }
    if (state.modelId !== lastModelId) {
      lastModelId = state.modelId;
      lastActiveId = null;
    }

    const activeBox = layout.positions.get(activeNode?.id);
    if (state.panTarget === "fit") {
      fitView();
    } else if (state.panTarget === "stack" && stackBox) {
      frameRect(stackBox);
    } else if (state.panTarget === "nodes") {
      const boxes = (state.focusIds || [])
        .map((id) => layout.positions.get(id))
        .filter(Boolean);
      if (boxes.length) {
        frameRect({
          minX: Math.min(...boxes.map((box) => box.x)) - 24,
          minY: Math.min(...boxes.map((box) => box.y)) - 36,
          maxX: Math.max(...boxes.map((box) => box.x + box.width)) + 24,
          maxY: Math.max(...boxes.map((box) => box.y + box.height)) + 24,
        });
      } else if (activeBox) {
        frameRect(boxRect(activeBox));
      }
    } else if (state.panTarget === "active" && activeBox) {
      frameRect(boxRect(activeBox));
    } else if (activeBox && state.autoScroll !== false
      && (playing || (lastActiveId !== null && activeNode.id !== lastActiveId))) {
      view.tx = cw / 2 - (activeBox.x + activeBox.width / 2) * view.scale;
      view.ty = ch / 2 - (activeBox.y + activeBox.height / 2) * view.scale;
    }
    if (activeNode) lastActiveId = activeNode.id;

    const paths = [];
    nodes.forEach((item) => {
      const target = layout.positions.get(item.id);
      item.dependsOn.filter((id) => layout.visibleIds.has(id)).forEach((dependencyId) => {
        const source = layout.positions.get(dependencyId);
        const parent = nodes.find((n) => n.id === dependencyId);
        const active = activeNode && (item.id === activeNode.id || activeUpstream.has(item.id))
          && (dependencyId === activeNode.id || activeUpstream.has(dependencyId));
        const cacheEdge = (item.cacheRole !== "none" || parent?.cacheRole !== "none") && (item.cacheRole === "readAppend" || parent?.cacheRole === "write" || item.cacheRole === "write");
        const startX = source.x + source.width / 2;
        const startY = source.y + source.height;
        const endX = target.x + target.width / 2;
        const endY = target.y;
        const cls = `walkthrough-edge${active ? " is-active" : ""}${active && playing ? " is-flow" : ""}${cacheEdge && mode === "decode" ? " is-cache" : ""}`;
        let d;
        if (target.row - source.row >= 2) {
          // Residual skip: bow to the side instead of crossing intermediate rows.
          const bowX = Math.min(startX, endX) > layout.axis
            ? layout.width - 26
            : 26;
          const midY = (startY + endY) / 2;
          d = `M ${startX} ${startY} C ${startX} ${startY + 50}, ${bowX} ${startY + 66}, ${bowX} ${midY}`
            + ` C ${bowX} ${endY - 66}, ${endX} ${endY - 50}, ${endX} ${endY}`;
          if (item.spine) {
            paths.push(`<g class="walkthrough-plus" transform="translate(${bowX} ${midY})"><circle r="7" /><text text-anchor="middle" dy="4">+</text></g>`);
          }
        } else {
          const control = Math.max(22, (endY - startY) * 0.45);
          d = `M ${startX} ${startY} C ${startX} ${startY + control}, ${endX} ${endY - control}, ${endX} ${endY}`;
        }
        paths.push(`<path class="${cls}" d="${d}" marker-end="url(#walkthroughArrow)" />`);
      });
    });

    const groupMarkup = [];
    const focusSet = new Set(state.focusIds || []);
    const blockGroups = [1, 2].filter((groupId) => groupBoxes.has(groupId));
    if (blockGroups.length) {
      // Frame around the repeated decoder block (attention + FFN).
      const boxes = blockGroups.map((groupId) => groupBoxes.get(groupId));
      const minX = Math.min(...boxes.map((b) => b.minX)) - 34;
      const minY = Math.min(...boxes.map((b) => b.minY)) - 44;
      const maxX = Math.max(...boxes.map((b) => b.maxX)) + 34;
      const maxY = Math.max(...boxes.map((b) => b.maxY)) + 18;
      groupMarkup.push(`<rect class="walkthrough-block-frame" x="${minX}" y="${minY}" width="${maxX - minX}" height="${maxY - minY}" rx="14" />`);
      groupMarkup.push(`<text class="walkthrough-block-label" x="${minX + 14}" y="${minY + 22}">DECODER LAYER · 代表层${state.layerNote ? ` · ${escapeHtml(state.layerNote)}` : ""}</text>`);
    }
    if (focusSet.size) {
      const focusBoxes = [...focusSet].map((id) => layout.positions.get(id)).filter(Boolean);
      if (focusBoxes.length) {
        const minX = Math.min(...focusBoxes.map((box) => box.x)) - 18;
        const minY = Math.min(...focusBoxes.map((box) => box.y)) - 28;
        const maxX = Math.max(...focusBoxes.map((box) => box.x + box.width)) + 18;
        const maxY = Math.max(...focusBoxes.map((box) => box.y + box.height)) + 18;
        groupMarkup.push(`<rect class="walkthrough-chapter-frame" x="${minX}" y="${minY}" width="${maxX - minX}" height="${maxY - minY}" rx="10" />`);
      }
    }
    const focusGroup = activeNode ? groupOf(activeNode) : null;
    groupBoxes.forEach((box, groupId) => {
      const faded = focusGroup !== null && groupId !== focusGroup ? " is-faded" : "";
      groupMarkup.push(`<text class="walkthrough-group-label${faded}" data-group-link="${groupId}" x="${box.minX - 12}" y="${box.minY - 10}">${GROUP_LABELS[groupId] || ""}</text>`);
      if (groupId !== 3 && blockGroups.includes(groupId)) {
        groupMarkup.push(`<line class="walkthrough-group-rule" x1="${box.minX - 12}" y1="${box.minY - 24}" x2="${box.maxX + 12}" y2="${box.minY - 24}" />`);
      }
    });

    const orderIndex = new Map(nodes.map((item, i) => [item.id, i]));
    const activeOrd = orderIndex.get(activeNode?.id) ?? 0;
    const nodeMarkup = nodes.map((item, index) => {
      const box = layout.positions.get(item.id);
      const g = box.geo;
      const color = kindColors[item.kind] || "#111";
      const selected = item.id === activeNode?.id;
      const upstream = activeUpstream.has(item.id);
      const inChapter = focusSet.has(item.id);
      const done = visited.has(item.id) && !selected;
      const future = playing && orderIndex.get(item.id) > activeOrd && !selected && !upstream;
      const pending = playing && !selected && !done && !upstream && !future;
      const dimmed = !playing && (
        focusSet.size ? !inChapter && !selected : activeNode && !selected && !upstream
      );
      const chapter = !playing && inChapter && !selected;
      const hot = item.id === hotId && !playing;
      const cls = `walkthrough-node tensor-${escapeHtml(item.kind)}${selected ? " is-selected" : ""}${upstream ? " is-upstream" : ""}${done ? " is-done" : ""}${pending ? " is-pending" : ""}${future ? " is-future" : ""}${dimmed ? " is-dimmed" : ""}${chapter ? " is-chapter" : ""}${hot ? " is-hot" : ""}`;
      const aria = escapeHtml(`${item.label}, ${resolveShape(item.shape, ctx)}`);

      const sheetCount = g.stack > 1 ? Math.min(3, g.stack - 1) : 0;
      const sheets = [];
      for (let k = sheetCount; k >= 1; k -= 1) {
        sheets.push(`<rect class="walkthrough-node-sheet" x="${box.x + 4 * k}" y="${box.y - 4 * k}" width="${box.width}" height="${box.height}" opacity="${(0.65 - 0.15 * k).toFixed(2)}" />`);
      }
      const stackLabel = g.stack > 1
        ? `<text class="walkthrough-dim" x="${box.x + box.width + 4 * sheetCount}" y="${box.y - 4 * sheetCount - 6}" text-anchor="end">×${fmtNum(g.stack)}</text>`
        : "";
      const nameY = box.y - 4 * sheetCount - 10;

      const tokens = (item.id === "token_ids" || item.id === "text_ids") ? tokenStrip(box) : "";
      const scanW = Math.max(8, box.width * 0.16);
      const scan = selected && playing && (g.matrix || tokens)
        ? `<rect class="walkthrough-scan" x="${box.x}" y="${box.y}" width="${scanW}" height="${box.height}"><animate attributeName="x" from="${box.x}" to="${box.x + box.width - scanW}" dur="0.7s" repeatCount="indefinite" /></rect>`
        : "";

      if (g.chip) {
        return `
        <g class="${cls}" data-node="${escapeHtml(item.id)}" role="button" tabindex="0" aria-label="${aria}">
          ${sheets.join("")}
          <rect class="walkthrough-node-face" x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}" rx="6" />
          <rect x="${box.x}" y="${box.y}" width="4" height="${box.height}" rx="2" fill="${color}" />
          ${tokens}${scan}
          ${stackLabel}
          <text class="walkthrough-node-index" x="${box.x + box.width - 8}" y="${box.y + 15}" text-anchor="end">${String(index + 1).padStart(2, "0")}</text>
          <text class="walkthrough-node-label" x="${box.x + 12}" y="${box.y + 27}">${escapeHtml(item.label)}</text>
        </g>`;
      }

      const grid = g.matrix
        ? `<rect x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}" fill="url(#tensorGrid)" rx="6" />`
        : "";
      const experts = item.stack && item.stackActive ? expertGrid(item, box) : "";
      return `
        <g class="${cls}" data-node="${escapeHtml(item.id)}" role="button" tabindex="0" aria-label="${aria}">
          ${sheets.join("")}
          <rect class="walkthrough-node-face" x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}" rx="6" />
          <rect x="${box.x}" y="${box.y}" width="4" height="${box.height}" rx="2" fill="${color}" />
          ${grid}
          ${experts}
          ${tokens}${scan}
          ${stackLabel}
          <text class="walkthrough-node-label" x="${box.x + box.width / 2}" y="${nameY}" text-anchor="middle">${escapeHtml(item.label)}</text>
          <text class="walkthrough-node-index" x="${box.x + box.width - 8}" y="${box.y + 16}" text-anchor="end">${String(index + 1).padStart(2, "0")}</text>
          <text class="walkthrough-dim" x="${box.x + box.width / 2}" y="${box.y + box.height + 14}" text-anchor="middle">${escapeHtml(fmtDim(g.cols))}</text>
          <text class="walkthrough-dim" transform="translate(${box.x - 10} ${box.y + box.height / 2}) rotate(-90)" text-anchor="middle">${escapeHtml(fmtDim(g.rows))}</text>
        </g>`;
    }).join("");

    svg.setAttribute("viewBox", `0 0 ${cw} ${ch}`);
    svg.innerHTML = `
      <defs>
        <marker id="walkthroughArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
        <pattern id="tensorGrid" width="12" height="10" patternUnits="userSpaceOnUse">
          <path d="M 12 0 L 0 0 0 10" fill="none" stroke="rgba(0,0,0,.10)" stroke-width="1" />
        </pattern>
      </defs>
      <g class="walkthrough-viewport" transform="translate(${view.tx} ${view.ty}) scale(${view.scale})">
        <g class="walkthrough-groups">${stackMarkup}${groupMarkup.join("")}</g>
        <g class="walkthrough-edges">${tokenMapLines(layout.positions, nodes)}${paths.join("")}</g>
        <g>${nodeMarkup}</g>
      </g>`;

    const panToBox = (box) => {
      if (!box) return;
      view.tx = cw / 2 - ((box.minX + box.maxX) / 2) * view.scale;
      view.ty = ch / 2 - box.minY * view.scale;
      applyView(svg);
    };
    svg.querySelectorAll("[data-group-link]").forEach((element) => {
      element.addEventListener("click", () => {
        const groupId = Number(element.dataset.groupLink);
        panToBox(groupBoxes.get(groupId));
        state.onSelectGroup?.(groupId);
      });
    });
    svg.querySelectorAll("[data-stack]").forEach((element) => {
      element.addEventListener("click", () => {
        if (blockFrameTop !== null) {
          view.ty = ch / 2 - blockFrameTop * view.scale;
          applyView(svg);
        }
        state.onChapter?.("layers");
      });
    });
    const viewport = svg.querySelector(".walkthrough-viewport");
    const clearFlow = () => viewport?.querySelector(".walkthrough-flow")?.remove();
    svg.querySelectorAll("[data-node]").forEach((element) => {
      const item = nodes.find((node) => node.id === element.dataset.node);
      const box = layout.positions.get(element.dataset.node);
      const select = () => {
        if (Date.now() < suppressClickUntil) return;
        onSelect?.(element.dataset.node);
      };
      element.addEventListener("click", select);
      element.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          select();
        }
      });
      element.addEventListener("pointerenter", () => {
        clearFlow();
        if (item && box && viewport) viewport.insertAdjacentHTML("beforeend", flowCardMarkup(item, box, ctx));
        element.classList.add("is-hover");
      });
      element.addEventListener("pointerleave", () => {
        clearFlow();
        element.classList.remove("is-hover");
      });
    });

    return { nodes, activeNode, scale: view.scale };
  }

  window.BallparkWalkthrough = {
    render, visibleNodes, resolveShape, upstreamIds, topologicalOrder, nodeGeometry, groupOf, GROUP_LABELS,
    layerSequence, representedLayerKind, remainingLayers,
  };
}());
