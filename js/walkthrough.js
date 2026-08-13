(function () {
  const kindColors = {
    activation: "#c9f36a",
    weight: "#5574ff",
    operation: "#a67cff",
    cache: "#ff8d5b",
    communication: "#ff806c",
    module: "#83b8ff",
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

  function computeLayout(nodes) {
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

    const nodeW = 150;
    const nodeH = 70;
    const colW = 186;
    const rowH = 104;
    const sideMargin = 96;
    const maxCols = Math.max(1, ...[...rows.values()].map((items) => items.length));
    const width = Math.max(560, sideMargin * 2 + maxCols * colW);
    const axis = width / 2;

    const orderedDepths = [...rows.keys()].sort((a, b) => a - b);
    const positions = new Map();
    let y = 26;
    let previousGroup = null;
    orderedDepths.forEach((itemDepth) => {
      const items = rows.get(itemDepth);
      const group = Math.min(...items.map((item) => GROUP_OF_STAGE[item.stageIndex] ?? 2));
      if (previousGroup !== null && group !== previousGroup) y += 58;
      previousGroup = group;
      const rowWidth = items.length * colW;
      items.forEach((item, index) => {
        positions.set(item.id, {
          x: axis - rowWidth / 2 + index * colW + (colW - nodeW) / 2,
          y,
          width: nodeW,
          height: nodeH,
          group,
        });
      });
      y += rowH;
    });
    const height = y + 24;

    const groupBoxes = new Map();
    positions.forEach((box) => {
      const current = groupBoxes.get(box.group) || {
        minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity,
      };
      current.minX = Math.min(current.minX, box.x);
      current.minY = Math.min(current.minY, box.y);
      current.maxX = Math.max(current.maxX, box.x + box.width);
      current.maxY = Math.max(current.maxY, box.y + box.height);
      groupBoxes.set(box.group, current);
    });

    return { width, height, positions, visibleIds, groupBoxes, axis };
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

  function fitScale(layout, zoom) {
    if (zoom !== "fit") return Math.max(0.2, Number(zoom) || 1);
    const scroller = document.querySelector(".walkthrough-stage-scroll");
    if (!scroller) return 1;
    return Math.min(
      1.1,
      Math.max(0.16, (scroller.clientWidth - 6) / layout.width),
    );
  }

  function matrixCells(item, box) {
    const color = kindColors[item.kind] || "#c9f36a";
    const cells = [];
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        cells.push(`<rect x="${box.x + 12 + column * 6}" y="${box.y + 10 + row * 6}" width="4" height="4" rx="0.8" fill="${color}" opacity="${0.4 + ((row + column) % 3) * 0.2}" />`);
      }
    }
    return cells.join("");
  }

  function render(state) {
    const svg = document.getElementById("walkthroughSvg");
    if (!svg || !state?.graph) return;
    const { graph, ep, branch, mode, ctx, activeId, hotId, onSelect, visitedIds, playing } = state;
    const nodes = visibleNodes(graph, { ep, branch });
    const activeNode = nodes.find((item) => item.id === activeId) || nodes[0];
    const layout = computeLayout(nodes);
    const activeUpstream = activeNode ? upstreamIds(nodes, activeNode.id) : new Set();
    const visited = visitedIds instanceof Set ? visitedIds : new Set();

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
        const cls = `walkthrough-edge${active ? " is-active" : ""}${cacheEdge && mode === "decode" ? " is-cache" : ""}`;
        let d;
        if (endY - startY > 150) {
          // Residual skip: bow to the side instead of crossing intermediate rows.
          const bowX = Math.min(startX, endX) > layout.axis
            ? layout.width - 26
            : 26;
          d = `M ${startX} ${startY} C ${startX} ${startY + 50}, ${bowX} ${startY + 66}, ${bowX} ${(startY + endY) / 2}`
            + ` C ${bowX} ${endY - 66}, ${endX} ${endY - 50}, ${endX} ${endY}`;
        } else {
          const control = Math.max(22, (endY - startY) * 0.45);
          d = `M ${startX} ${startY} C ${startX} ${startY + control}, ${endX} ${endY - control}, ${endX} ${endY}`;
        }
        paths.push(`<path class="${cls}" d="${d}" marker-end="url(#walkthroughArrow)" />`);
      });
    });

    const groupMarkup = [];
    const blockGroups = [1, 2].filter((groupId) => layout.groupBoxes.has(groupId));
    if (blockGroups.length) {
      // Frame around the repeated decoder block (attention + FFN).
      const boxes = blockGroups.map((groupId) => layout.groupBoxes.get(groupId));
      const minX = Math.min(...boxes.map((b) => b.minX)) - 34;
      const minY = Math.min(...boxes.map((b) => b.minY)) - 44;
      const maxX = Math.max(...boxes.map((b) => b.maxX)) + 34;
      const maxY = Math.max(...boxes.map((b) => b.maxY)) + 18;
      groupMarkup.push(`<rect class="walkthrough-block-frame" x="${minX}" y="${minY}" width="${maxX - minX}" height="${maxY - minY}" rx="14" />`);
      groupMarkup.push(`<text class="walkthrough-block-label" x="${minX + 14}" y="${minY + 22}">DECODER LAYER${state.layerNote ? ` · ${escapeHtml(state.layerNote)}` : ""}</text>`);
    }
    layout.groupBoxes.forEach((box, groupId) => {
      if (groupId === 0) return;
      groupMarkup.push(`<text class="walkthrough-group-label" x="${box.minX - 12}" y="${box.minY - 10}">${GROUP_LABELS[groupId] || ""}</text>`);
      if (groupId !== 3 && blockGroups.includes(groupId)) {
        groupMarkup.push(`<line class="walkthrough-group-rule" x1="${box.minX - 12}" y1="${box.minY - 24}" x2="${box.maxX + 12}" y2="${box.minY - 24}" />`);
      }
    });

    const nodeMarkup = nodes.map((item, index) => {
      const box = layout.positions.get(item.id);
      const selected = item.id === activeNode?.id;
      const upstream = activeUpstream.has(item.id);
      const done = visited.has(item.id) && !selected;
      const pending = playing && !selected && !done && !upstream;
      const dimmed = !playing && activeNode && !selected && !upstream;
      const hot = item.id === hotId && !playing;
      return `
        <g class="walkthrough-node tensor-${escapeHtml(item.kind)}${selected ? " is-selected" : ""}${upstream ? " is-upstream" : ""}${done ? " is-done" : ""}${pending ? " is-pending" : ""}${dimmed ? " is-dimmed" : ""}${hot ? " is-hot" : ""}"
           data-node="${escapeHtml(item.id)}" role="button" tabindex="0"
           aria-label="${escapeHtml(`${item.label}, ${resolveShape(item.shape, ctx)}`)}">
          <rect class="walkthrough-node-back" x="${box.x + 4}" y="${box.y + 4}" width="${box.width}" height="${box.height}" rx="6" />
          <rect class="walkthrough-node-face" x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}" rx="6" />
          <rect x="${box.x}" y="${box.y}" width="4" height="${box.height}" rx="2" fill="${kindColors[item.kind] || "#c9f36a"}" />
          ${matrixCells(item, box)}
          <text class="walkthrough-node-index" x="${box.x + box.width - 8}" y="${box.y + 16}" text-anchor="end">${String(index + 1).padStart(2, "0")}</text>
          <text class="walkthrough-node-label" x="${box.x + 12}" y="${box.y + 44}">${escapeHtml(item.label)}</text>
          <text class="walkthrough-node-shape" x="${box.x + 12}" y="${box.y + 59}">${escapeHtml(resolveShape(item.shape, ctx))}</text>
        </g>`;
    }).join("");

    const scale = fitScale(layout, state.zoom);
    svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
    svg.setAttribute("preserveAspectRatio", "xMinYMin meet");
    svg.style.width = `${layout.width * scale}px`;
    svg.style.height = `${layout.height * scale}px`;
    svg.innerHTML = `
      <defs>
        <marker id="walkthroughArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>
      <g class="walkthrough-groups">${groupMarkup.join("")}</g>
      <g class="walkthrough-edges">${paths.join("")}</g>
      <g>${nodeMarkup}</g>`;

    svg.querySelectorAll("[data-node]").forEach((element) => {
      const select = () => onSelect?.(element.dataset.node);
      element.addEventListener("click", select);
      element.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select();
        }
      });
    });

    const activeBox = layout.positions.get(activeNode?.id);
    if (activeBox && state.autoScroll !== false) {
      window.requestAnimationFrame(() => {
        const header = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-h")) || 0;
        const drawer = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--drawer-h")) || 0;
        const nodeY = (activeBox.y + activeBox.height / 2) * scale;
        const svgTop = svg.getBoundingClientRect().top + window.scrollY;
        const viewH = Math.max(160, window.innerHeight - header - drawer);
        window.scrollTo({
          top: Math.max(0, svgTop + nodeY - header - viewH / 2),
          behavior: playing ? "auto" : "smooth",
        });
      });
    }

    return { nodes, activeNode, scale };
  }

  window.BallparkWalkthrough = { render, visibleNodes, resolveShape, upstreamIds, topologicalOrder };
}());
