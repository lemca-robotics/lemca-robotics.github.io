// Interactive sensor visualization for point-to-disk task — uses D3 v7
(function () {
  const CONTAINER_ID = "p2d-sensors";
  const R_MAX = 2.0;
  const ARENA_R = 1.0;
  const TARGET_R = 0.01;
  const WORLD = 1.3;
  const SIZE = 260;

  const COST_CONFIGS = {
    costly: {
      label: "CostlyGPS", r_max: 2.0, mem: 0.005,
      gps: { base: 0.01, scale: 0.01 },
      sector_radar: { base: 0.01, rad_scale: 0.01, dist_scale: 0.01 },
    },
    cheap: {
      label: "CheapGPS", r_max: 2.0, mem: 0.005,
      gps: { base: 0.02, scale: 0.01 },
      sector_radar: { base: 0.01, rad_scale: 0.05, dist_scale: 0.01 },
    },
  };

  // ── Cost math ─────────────────────────────────────────────────────────────
  function bits(span, sigma) { return Math.max(Math.log2(span / (sigma + 1e-6)), 0); }

  function computeCost(cfg, sensorType, params) {
    const memCost = params.history > 1 ? cfg.mem * Math.log2(params.history) : 0;
    if (sensorType === "gps") {
      return cfg.gps.base + cfg.gps.scale * 2 * bits(2 * R_MAX, params.sigma_xy) + memCost;
    }
    const s = params.s;
    const w = (2 * Math.PI) / s;
    const Rsinc = Math.sin(w / 2) / (w / 2 + 1e-9);
    const sigma_theta = Math.sqrt(-2 * Math.log(Math.max(Rsinc, 1e-9)) + params.sigma_theta ** 2);
    const rad_cost = cfg.sector_radar.rad_scale * bits(2 * Math.PI, sigma_theta);
    let dist_cost = 0;
    if (params.d.length > 0) {
      const bands = [0, ...params.d.slice().sort((a, b) => a - b), cfg.r_max];
      let sq = 0;
      for (let i = 0; i < bands.length - 1; i++) {
        const prob = (bands[i + 1] ** 2 - bands[i] ** 2) / cfg.r_max ** 2;
        sq += prob * ((bands[i + 1] - bands[i]) ** 2 / 12);
      }
      dist_cost = cfg.sector_radar.dist_scale * bits(cfg.r_max, Math.sqrt(sq + params.sigma_r ** 2));
    }
    return cfg.sector_radar.base + rad_cost + dist_cost + memCost;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function randn() {
    let u, v;
    do { u = Math.random(); } while (u === 0);
    do { v = Math.random(); } while (v === 0);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function parseD(str) {
    return (str || "").split(",").map(s => parseFloat(s.trim())).filter(v => !isNaN(v) && v > 0 && v < R_MAX);
  }

  // ── Coord helpers ─────────────────────────────────────────────────────────
  const scX = d3.scaleLinear().domain([-WORLD, WORLD]).range([0, SIZE]);
  const scY = d3.scaleLinear().domain([-WORLD, WORLD]).range([SIZE, 0]);
  const px = r => r / (2 * WORLD) * SIZE;
  const arc = d3.arc();

  // ── Colors ────────────────────────────────────────────────────────────────
  function cellHue(si, s) { return (360 * si / s) % 360; }
  function cellColor(si, bi, s, numBands, isActive) {
    const hue = cellHue(si, s);
    if (isActive) return `hsla(${hue}, 80%, 42%, 0.85)`;
    const light = numBands <= 1 ? 75 : 62 + (bi / (numBands - 1)) * 20;
    return `hsla(${hue}, 55%, ${light.toFixed(0)}%, 0.45)`;
  }

  // ── Sector geometry ───────────────────────────────────────────────────────
  function buildCells(params) {
    const { s, phi, d: rawD } = params;
    const sorted_d = rawD.slice().sort((a, b) => a - b);
    const bands = [0, ...sorted_d, WORLD];
    const secWidth = (2 * Math.PI) / s;
    const half = secWidth / 2;
    const cells = [];
    for (let bi = 0; bi < bands.length - 1; bi++) {
      for (let si = 0; si < s; si++) {
        const center = phi + si * secWidth;
        cells.push({ si, bi, r0: bands[bi], r1: Math.min(bands[bi + 1], WORLD), a0: center - half, a1: center + half });
      }
    }
    return cells;
  }

  function activeCellFor(wx, wy, params) {
    const { s, phi, d: rawD } = params;
    const r = Math.hypot(wx, wy);
    const theta = Math.atan2(wy, wx);
    const secWidth = (2 * Math.PI) / s;
    const normalized = ((theta - phi + secWidth / 2) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
    const bestSec = Math.min(Math.floor(normalized / secWidth), s - 1);
    const sorted_d = rawD.slice().sort((a, b) => a - b);
    const bands = [0, ...sorted_d, R_MAX];
    let bestBand = bands.length - 2;
    for (let i = 0; i < bands.length - 1; i++) {
      if (r >= bands[i] && r < bands[i + 1]) { bestBand = i; break; }
    }
    return { si: bestSec, bi: bestBand };
  }

  // ── Arena base (shared) ───────────────────────────────────────────────────
  function drawArena(svg) {
    svg.append("line").attr("x1", 0).attr("x2", SIZE).attr("y1", scY(0)).attr("y2", scY(0)).style("stroke", "#ddd").style("stroke-width", 0.5);
    svg.append("line").attr("x1", scX(0)).attr("x2", scX(0)).attr("y1", 0).attr("y2", SIZE).style("stroke", "#ddd").style("stroke-width", 0.5);
    svg.append("circle").attr("cx", scX(0)).attr("cy", scY(0)).attr("r", px(ARENA_R)).style("fill", "none").style("stroke", "#999").style("stroke-width", 1).style("stroke-dasharray", "4 4");
    svg.append("circle").attr("cx", scX(0)).attr("cy", scY(0)).attr("r", px(TARGET_R)).style("fill", "#111");
  }

  // ── Slider / input helpers ────────────────────────────────────────────────
  function addSlider(parent, { label, id, min, max, step, value, fmt }, onChange) {
    const row = parent.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "1px").style("margin-bottom", "4px");
    const top = row.append("div").style("display", "flex").style("justify-content", "space-between").style("font-size", "12px");
    top.append("span").text(label);
    const valSpan = top.append("span").attr("id", id + "-val").style("font-weight", "600").text((fmt || (v => v))(value));
    row.append("input").attr("type", "range").attr("id", id).attr("min", min).attr("max", max).attr("step", step).attr("value", value)
      .style("width", "100%")
      .on("input", function () { valSpan.text((fmt || (v => v))(this.value)); onChange(); });
  }

  function addTextInput(parent, { label, id, value }, onChange) {
    const row = parent.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "1px").style("margin-bottom", "4px");
    row.append("label").attr("for", id).style("font-size", "12px").text(label);
    row.append("input").attr("type", "text").attr("id", id).attr("value", value)
      .style("font-size", "11px").style("padding", "3px 5px").style("border", "1px solid #ccc").style("border-radius", "3px").style("width", "100%").style("box-sizing", "border-box")
      .on("input", onChange);
  }

  function val(id) { const el = document.getElementById(id); return el ? el.value : ""; }

  // ── Main init ─────────────────────────────────────────────────────────────
  function init() {
    const container = d3.select("#" + CONTAINER_ID);
    if (container.empty()) return;
    container.html("");

    let rsClickedPt = null;
    let gpsClickedPt = null;

    // ── Outer: left controls | right viz ──
    const outer = container.append("div")
      .style("display", "flex").style("gap", "20px").style("align-items", "flex-start");

    // Left panel: RS controls only
    const leftPanel = outer.append("div")
      .style("width", "190px").style("flex-shrink", "0").style("display", "flex").style("flex-direction", "column").style("gap", "6px");

    leftPanel.append("div").style("font-weight", "700").style("font-size", "13px").style("color", "#333").text("RadialSector");
    const rsControls = leftPanel.append("div");

    // Right panel: [RS col] [mid col] [GPS col]
    const rightPanel = outer.append("div")
      .style("display", "flex").style("gap", "10px").style("align-items", "flex-start").style("flex", "1");

    // RS column: cost header → SVG → click info
    const rsCol = rightPanel.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "4px");
    const rsCostDiv = rsCol.append("div")
      .style("font-size", "11px").style("line-height", "1.5").style("min-height", "36px")
      .style("background", "#f8f8f8").style("border", "1px solid #e8e8e8").style("border-radius", "4px").style("padding", "4px 8px");
    const rsSvg = rsCol.append("svg").attr("width", SIZE).attr("height", SIZE)
      .style("border", "1px solid #ddd").style("border-radius", "4px").style("cursor", "crosshair").style("display", "block");
    const rsInfo = rsCol.append("div").style("font-size", "10px").style("color", "#888").style("min-height", "13px");

    // Middle column: diff panel, self-centered
    const midCol = rightPanel.append("div")
      .style("display", "flex").style("flex-direction", "column").style("gap", "6px")
      .style("align-self", "center").style("width", "115px").style("flex-shrink", "0");
    midCol.append("div")
      .style("font-size", "10px").style("font-weight", "700").style("color", "#555")
      .style("text-align", "center").style("text-transform", "uppercase").style("letter-spacing", "0.05em")
      .text("Δ RS − GPS");
    const diffPanel = midCol.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "5px");

    // GPS section: [cost+SVG+info col] [controls col]
    const gpsSec = rightPanel.append("div").style("display", "flex").style("gap", "12px").style("align-items", "flex-start");

    const gpsCol = gpsSec.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "4px");
    const gpsCostDiv = gpsCol.append("div")
      .style("font-size", "11px").style("line-height", "1.5").style("min-height", "36px")
      .style("background", "#f8f8f8").style("border", "1px solid #e8e8e8").style("border-radius", "4px").style("padding", "4px 8px");
    const gpsSvg = gpsCol.append("svg").attr("width", SIZE).attr("height", SIZE)
      .style("border", "1px solid #ddd").style("border-radius", "4px").style("cursor", "crosshair").style("display", "block");
    const gpsInfo = gpsCol.append("div").style("font-size", "10px").style("color", "#888").style("min-height", "13px");

    // GPS controls to the right of the GPS SVG
    const gpsCtrlWrap = gpsSec.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "6px");
    gpsCtrlWrap.append("div").style("font-weight", "700").style("font-size", "13px").style("color", "#333").text("GPS");
    const gpsControls = gpsCtrlWrap.append("div").style("width", "170px");

    // ── SVG layers ──
    const rsLayers = { cells: rsSvg.append("g"), arena: rsSvg.append("g"), noise: rsSvg.append("g"), point: rsSvg.append("g") };
    const gpsLayers = { cells: gpsSvg.append("g"), arena: gpsSvg.append("g"), noise: gpsSvg.append("g"), point: gpsSvg.append("g") };
    drawArena(rsLayers.arena);
    drawArena(gpsLayers.arena);

    // ── Click handlers — mirrored ──
    function handleClick(wx, wy) {
      if (Math.hypot(wx, wy) > ARENA_R) {
        rsClickedPt = null; gpsClickedPt = null;
        rsInfo.text(""); gpsInfo.text("");
      } else {
        rsClickedPt = [wx, wy]; gpsClickedPt = [wx, wy];
        const label = `(${wx.toFixed(3)}, ${wy.toFixed(3)})  r=${Math.hypot(wx, wy).toFixed(3)}`;
        rsInfo.text(label); gpsInfo.text(label);
      }
      redraw();
    }

    rsSvg.on("click", function (event) {
      const [cx, cy] = d3.pointer(event);
      handleClick(scX.invert(cx), scY.invert(cy));
    });

    gpsSvg.on("click", function (event) {
      const [cx, cy] = d3.pointer(event);
      handleClick(scX.invert(cx), scY.invert(cy));
    });

    // ── Parameter getters ──
    function getRSParams() {
      return {
        s: +val("rs-s") || 4,
        phi: +val("rs-phi") || 0,
        sigma_theta: +val("rs-st") || 0,
        sigma_r: +val("rs-sr") || 0,
        d: parseD(val("rs-d")),
        history: +val("rs-hist") || 1,
      };
    }

    function getGPSParams() {
      return { sigma_xy: +val("gps-sigma") || 0, history: +val("gps-hist") || 1 };
    }

    // ── Draw RadialSector ──
    function drawRS(params) {
      const cells = buildCells(params);
      const active = rsClickedPt ? activeCellFor(rsClickedPt[0], rsClickedPt[1], params) : null;
      const numBands = params.d.length + 1;

      rsLayers.cells.selectAll("path").data(cells, d => `${d.si}-${d.bi}`)
        .join("path")
        .attr("d", d => arc({ innerRadius: px(d.r0), outerRadius: px(d.r1), startAngle: -(d.a1 - Math.PI / 2), endAngle: -(d.a0 - Math.PI / 2) }))
        .attr("transform", `translate(${scX(0)},${scY(0)})`)
        .style("fill", d => {
          const isActive = active && d.si === active.si && d.bi === active.bi;
          return cellColor(d.si, d.bi, params.s, numBands, isActive);
        })
        .style("stroke", d => {
          const isActive = active && d.si === active.si && d.bi === active.bi;
          return isActive ? `hsl(${cellHue(d.si, params.s)}, 70%, 28%)` : "rgba(100,100,100,0.25)";
        })
        .style("stroke-width", d => active && d.si === active.si && d.bi === active.bi ? 2 : 0.8);

      rsLayers.noise.selectAll("*").remove();
      if (rsClickedPt && (params.sigma_r > 0 || params.sigma_theta > 0)) {
        const [wx, wy] = rsClickedPt;
        const r = Math.hypot(wx, wy), theta = Math.atan2(wy, wx);
        const pts = d3.range(300).map(() => {
          const nr = r + params.sigma_r * randn();
          const nt = theta + params.sigma_theta * randn();
          return [nr * Math.cos(nt), nr * Math.sin(nt)];
        });
        rsLayers.noise.selectAll("circle").data(pts).join("circle")
          .attr("cx", d => scX(d[0])).attr("cy", d => scY(d[1])).attr("r", 2)
          .style("fill", "rgba(200,50,50,0.45)").style("pointer-events", "none");
      }

      rsLayers.point.selectAll("*").remove();
      if (rsClickedPt) {
        rsLayers.point.append("circle").attr("cx", scX(rsClickedPt[0])).attr("cy", scY(rsClickedPt[1])).attr("r", 5).style("fill", "rgba(180,0,0,0.9)");
      }
    }

    // ── Draw GPS ──
    function drawGPS(params) {
      gpsLayers.cells.selectAll("*").remove();
      gpsLayers.noise.selectAll("*").remove();
      gpsLayers.point.selectAll("*").remove();
      if (!gpsClickedPt) return;
      const [wx, wy] = gpsClickedPt;
      const cx = scX(wx), cy = scY(wy);
      const s = params.sigma_xy;

      if (s > 0) {
        [[2, 0.07], [1, 0.15]].forEach(([mult, alpha]) => {
          gpsLayers.noise.append("circle").attr("cx", cx).attr("cy", cy).attr("r", px(s * mult))
            .style("fill", `rgba(99,150,237,${alpha})`).style("stroke", "rgba(50,100,200,0.5)").style("stroke-width", 1).style("pointer-events", "none");
        });
        const pts = d3.range(200).map(() => [wx + s * randn(), wy + s * randn()]);
        gpsLayers.noise.selectAll("circle.sample").data(pts).join("circle")
          .attr("class", "sample").attr("cx", d => scX(d[0])).attr("cy", d => scY(d[1])).attr("r", 2)
          .style("fill", "rgba(50,100,220,0.35)").style("pointer-events", "none");
      }
      gpsLayers.point.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 5).style("fill", "rgba(30,80,200,0.9)");
    }

    // ── Render diff panel ──
    function renderDiff(rsParams, gpsParams) {
      diffPanel.selectAll("*").remove();
      Object.values(COST_CONFIGS).forEach(cfg => {
        const diff = computeCost(cfg, "sector_radar", rsParams) - computeCost(cfg, "gps", gpsParams);
        const intensity = Math.min(Math.abs(diff) / 0.3, 1);
        const pos = diff >= 0;
        const bg = pos ? `rgba(34,139,34,${0.15 + intensity * 0.55})` : `rgba(200,50,50,${0.15 + intensity * 0.55})`;
        const fg = pos ? "#1a5c1a" : "#8b1a1a";
        const box = diffPanel.append("div")
          .style("background", bg).style("border-radius", "4px").style("padding", "5px 8px");
        box.append("div").style("font-size", "10px").style("color", fg).style("font-weight", "600").text(cfg.label);
        box.append("div").style("font-size", "13px").style("font-weight", "700").style("color", fg).style("text-align", "center")
          .text((diff >= 0 ? "+" : "") + diff.toFixed(4));
      });
    }

    // ── Render costs above each SVG ──
    function renderCosts(rsParams, gpsParams) {
      rsCostDiv.html(Object.values(COST_CONFIGS).map(cfg =>
        `<span style="color:#555">${cfg.label}:</span> <b>${computeCost(cfg, "sector_radar", rsParams).toFixed(4)}</b>`
      ).join("<br>"));
      gpsCostDiv.html(Object.values(COST_CONFIGS).map(cfg =>
        `<span style="color:#555">${cfg.label}:</span> <b>${computeCost(cfg, "gps", gpsParams).toFixed(4)}</b>`
      ).join("<br>"));
    }

    // ── Redraw all ──
    function redraw() {
      const rsP = getRSParams(), gpsP = getGPSParams();
      drawRS(rsP);
      drawGPS(gpsP);
      renderDiff(rsP, gpsP);
      renderCosts(rsP, gpsP);
    }

    // ── Build RS controls ──
    addSlider(rsControls, { label: "Sectors (s)", id: "rs-s", min: 1, max: 36, step: 1, value: 4, fmt: v => +v }, redraw);
    addSlider(rsControls, { label: "φ offset (rad)", id: "rs-phi", min: 0, max: 6.28, step: 0.05, value: 0, fmt: v => (+v).toFixed(2) }, redraw);
    addSlider(rsControls, { label: "σ_θ", id: "rs-st", min: 0, max: 0.8, step: 0.01, value: 0, fmt: v => (+v).toFixed(2) }, redraw);
    addSlider(rsControls, { label: "σ_r", id: "rs-sr", min: 0, max: 0.1, step: 0.005, value: 0, fmt: v => (+v).toFixed(3) }, redraw);
    addTextInput(rsControls, { label: "Distance thresholds d (comma-sep.)", id: "rs-d", value: "0.1, 0.5" }, redraw);
    addSlider(rsControls, { label: "History", id: "rs-hist", min: 1, max: 8, step: 1, value: 1, fmt: v => +v }, redraw);

    // ── Build GPS controls ──
    addSlider(gpsControls, { label: "σ_xy", id: "gps-sigma", min: 0, max: 0.5, step: 0.005, value: 0.05, fmt: v => (+v).toFixed(3) }, redraw);
    addSlider(gpsControls, { label: "History", id: "gps-hist", min: 1, max: 8, step: 1, value: 1, fmt: v => +v }, redraw);

    redraw();
  }

  function tryInit() {
    if (typeof d3 !== "undefined") { init(); }
    else { setTimeout(tryInit, 50); }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", tryInit);
  else tryInit();
})();
