// Interactive rangefinder noise-model visualization for the RF-DMC task — D3 v7.
// Ports lead/tasks/rs/base.py {RFSensor.__call__, RayScanSensor sensor grades,
// SensorCost} to the browser.
//
// Left: the beam range-finder mixture model p(z | z*) for the selected sensor
// grade, as both the analytic mixture density and a Monte-Carlo histogram, with
// z* on a slider. Right: the per-step sensor cost as a function of grade,
// nrays, and history (the SensorCost model).
(function () {
  const CONTAINER_ID = "rf-dmc-noise";

  // ── Sensor range (RFSensor / RayScanSensor) ──────────────────────────────
  const Z_MIN = 0.0, Z_MAX = 10.0;
  const LAMBDA_SHORT = 0.5;          // RFSensor.lambda_short default
  const N_SAMPLES = 5000;            // Monte-Carlo draws per re-sample
  const SIG_THR = 0.03;              // below this, the hit lobe renders as an atom

  // ── Sensor grades (RayScanSensor.__post_init__ sensor_grades) ────────────
  // c_short = 0 for every grade, so the short lobe is part of the model but
  // carries no weight; we still surface it in the legend for completeness.
  const GRADES = {
    F: { c_hit: 0.65,  c_short: 0, c_max: 0.10,  c_rand: 0.25,  sigma_hit: 0.10 },
    E: { c_hit: 0.85,  c_short: 0, c_max: 0.05,  c_rand: 0.10,  sigma_hit: 0.05 },
    D: { c_hit: 0.95,  c_short: 0, c_max: 0.01,  c_rand: 0.04,  sigma_hit: 0.01 },
    C: { c_hit: 0.97,  c_short: 0, c_max: 0.01,  c_rand: 0.02,  sigma_hit: 0.001 },
    B: { c_hit: 0.99,  c_short: 0, c_max: 0.005, c_rand: 0.005, sigma_hit: 0.0008 },
    A: { c_hit: 0.995, c_short: 0, c_max: 0.003, c_rand: 0.002, sigma_hit: 1e-5 },
  };
  const GRADE_ORDER = ["A", "B", "C", "D", "E", "F"]; // best → worst

  // ── Cost model (SensorCost) ──────────────────────────────────────────────
  // grade_costs are per-ray energy multiples (normalized to grade F = 1.0),
  // precomputed from RFSensor.report_sensor_stats; mem weights the history term.
  const GRADE_COST = { A: 22.609, B: 15.387, C: 7.615, D: 5.080, E: 2.444, F: 1.000 };
  const MEM = 0.5;
  const NRAYS = [1, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360];
  const HISTORY = [1, 2, 4, 6, 8, 16];
  function sensorCost(q, nrays, history) {
    const quality = GRADE_COST[q] * nrays;
    const mem = MEM * Math.log2(history) * nrays;
    return { quality, mem, perRay: GRADE_COST[q] + MEM * Math.log2(history), total: quality + mem };
  }

  // ── Palette (mixture components) ─────────────────────────────────────────
  const C = {
    bg: "#FAFAFA", grid: "#E5E5E5", axis: "#999",
    hit: "#2E86AB", short: "#5BA85B", max: "#E0A458", rand: "#9AA1A8",
    total: "#222", hist: "#cdd8df", zstar: "#1B4965",
  };

  // ── Math helpers ─────────────────────────────────────────────────────────
  const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
      a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return s * y;
  }
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
  const normPdf = (x, mu, s) => Math.exp(-0.5 * Math.pow((x - mu) / s, 2)) / (s * Math.sqrt(2 * Math.PI));
  // Thrun-normalized short lobe over [0, z*] (drawn for completeness; weight = 0).
  function shortPdf(x, zstar, lam) {
    if (x < 0 || x > zstar || zstar <= 0) return 0;
    const eta = 1 / (1 - Math.exp(-lam * zstar));
    return eta * lam * Math.exp(-lam * x);
  }

  // ── Beam model: single measurement draw (≈ RFSensor.__call__) ────────────
  function sampleOne(zstar, g) {
    const r = Math.random();
    if (r < g.c_hit) return clip(zstar + g.sigma_hit * randn(), Z_MIN, Z_MAX);
    if (r < g.c_hit + g.c_short) return clip(-Math.log(1 - Math.random()) / LAMBDA_SHORT, Z_MIN, zstar);
    if (r < g.c_hit + g.c_short + g.c_max) return Z_MAX;
    return Z_MIN + Math.random() * (Z_MAX - Z_MIN);
  }
  function sampleMany(zstar, g, n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = sampleOne(zstar, g);
    return out;
  }

  // ── Analytic continuous density (no atoms) and atom masses ───────────────
  function densCont(x, zstar, g) {
    let d = g.c_rand / (Z_MAX - Z_MIN);
    if (g.sigma_hit >= SIG_THR) d += g.c_hit * normPdf(x, zstar, g.sigma_hit);
    d += g.c_short * shortPdf(x, zstar, LAMBDA_SHORT);
    return d;
  }
  // Discrete probability atoms: z_max delta + the clipped-normal boundary mass
  // (or the whole hit lobe collapsed onto z* when σ_hit is below resolution).
  function atomList(zstar, g) {
    const A = [];
    let massMax = g.c_max, massMin = 0;
    if (g.sigma_hit >= SIG_THR) {
      massMax += g.c_hit * (1 - Phi((Z_MAX - zstar) / g.sigma_hit));
      massMin += g.c_hit * Phi((Z_MIN - zstar) / g.sigma_hit);
    } else {
      A.push({ z: zstar, mass: g.c_hit, color: C.hit, label: "hit" });
    }
    A.push({ z: Z_MAX, mass: massMax, color: C.max, label: "max" });
    if (massMin > 2e-3) A.push({ z: Z_MIN, mass: massMin, color: C.hit, label: "" });
    return A;
  }

  // ── DOM helpers ──────────────────────────────────────────────────────────
  function addSlider(parent, { label, id, min, max, step, value, fmt }, onChange) {
    const row = parent.append("div").style("display", "flex").style("flex-direction", "column")
      .style("gap", "1px").style("margin-bottom", "6px");
    const top = row.append("div").style("display", "flex").style("justify-content", "space-between")
      .style("font-size", "12px");
    top.append("span").html(label);
    const valSpan = top.append("span").style("font-weight", "600").text((fmt || (v => v))(value));
    row.append("input").attr("type", "range").attr("id", id).attr("min", min).attr("max", max)
      .attr("step", step).attr("value", value).style("width", "100%")
      .on("input", function () { valSpan.text((fmt || (v => v))(this.value)); onChange(); });
  }
  const val = id => { const el = document.getElementById(id); return el ? +el.value : 0; };

  // ── Main ─────────────────────────────────────────────────────────────────
  function init() {
    const container = d3.select("#" + CONTAINER_ID);
    if (container.empty()) return;
    container.html("");

    let grade = "D";
    let samples = null;

    const outer = container.append("div")
      .style("display", "flex").style("gap", "20px").style("align-items", "flex-start").style("flex-wrap", "wrap");

    // ── Left: controls ──
    const left = outer.append("div").style("width", "220px").style("flex-shrink", "0")
      .style("display", "flex").style("flex-direction", "column").style("gap", "7px");

    left.append("div").style("font-weight", "700").style("font-size", "13px").style("color", "#333")
      .text("Sensor grade");
    const gradeRow = left.append("div").style("display", "flex").style("gap", "4px");
    const gradeBtns = {};
    GRADE_ORDER.forEach(q => {
      gradeBtns[q] = gradeRow.append("button").text(q)
        .style("flex", "1").style("font-size", "12px").style("font-weight", "600")
        .style("padding", "6px 0").style("cursor", "pointer").style("border", "1px solid #bbb")
        .style("border-radius", "5px").style("background", "#fff").style("color", "#222")
        .on("click", () => { grade = q; resample(); drawAll(); });
    });
    // grade parameter readout
    const gradeInfo = left.append("div")
      .style("background", "#f8f8f8").style("border", "1px solid #e8e8e8").style("border-radius", "5px")
      .style("padding", "6px 8px").style("font-size", "10.5px").style("color", "#555").style("line-height", "1.5");

    left.append("div").style("font-weight", "700").style("font-size", "13px").style("color", "#333")
      .style("margin-top", "4px").text("True range z*");
    const ctl = left.append("div");

    const resampleBtn = left.append("button").html("⚡ Re-sample measurements<br/>(N = 5000)")
      .style("font-size", "12px").style("padding", "6px 8px").style("cursor", "pointer")
      .style("border", "1px solid #bbb").style("border-radius", "5px").style("background", "#fff").style("color", "#222");

    // ── empirical stats ──
    left.append("div").style("font-weight", "700").style("font-size", "13px").style("color", "#333")
      .style("margin-top", "5px").text("Measurement statistics");
    const statsPanel = left.append("div")
      .style("background", "#f8f8f8").style("border", "1px solid #e8e8e8").style("border-radius", "5px")
      .style("padding", "7px 9px").style("display", "flex").style("flex-direction", "column").style("gap", "4px");

    // ── Center: distribution plot (legend sits above the plot, outside it) ──
    // flex:1 + min-width:0 lets this column absorb all space left by the two
    // fixed side columns; the SVG width is measured from it (see layout()).
    const distCol = outer.append("div").style("flex", "1 1 360px").style("min-width", "0")
      .style("display", "flex").style("flex-direction", "column").style("gap", "4px");
    distCol.append("div").style("font-size", "12px").style("font-weight", "700").style("color", "#333")
      .html("Beam model &nbsp; <span style='font-weight:400;color:#888'>p(z &nbsp;|&nbsp; z*)</span>");
    const legendDiv = distCol.append("div").style("display", "flex").style("flex-wrap", "wrap")
      .style("gap", "10px").style("align-items", "center").style("padding", "2px 2px 4px")
      .style("font-size", "10.5px").style("color", "#444");
    let W = 520; const H = 333, M = { t: 16, r: 14, b: 36, l: 46 };
    const distSvg = distCol.append("svg").attr("width", W).attr("height", H)
      .style("border", "1px solid #ddd").style("border-radius", "4px").style("display", "block").style("background", C.bg);

    // ── Right: cost (sliders stacked, energy cost panel below them) ──
    const costCol = outer.append("div").style("width", "212px").style("flex-shrink", "0")
      .style("display", "flex").style("flex-direction", "column").style("gap", "7px");
    costCol.append("div").style("font-size", "12px").style("font-weight", "700").style("color", "#333")
      .text("Sensor cost / step");
    const costNrays = costCol.append("div");
    const costHist = costCol.append("div");
    const costPanel = costCol.append("div")
      .style("background", "#f8f8f8").style("border", "1px solid #e8e8e8").style("border-radius", "5px")
      .style("padding", "8px 10px").style("display", "flex").style("flex-direction", "column").style("gap", "6px");

    // ── Distribution plot drawing ──
    const gAxes = distSvg.append("g");
    const gHist = distSvg.append("g");
    const gComp = distSvg.append("g");
    const gTotal = distSvg.append("g");
    const gAtom = distSvg.append("g");
    const gMark = distSvg.append("g");

    function drawDist() {
      const g = GRADES[grade];
      const zstar = val("zstar");
      const x = d3.scaleLinear().domain([Z_MIN, Z_MAX]).range([M.l, W - M.r]);

      // histogram (density-normalized) of the Monte-Carlo draws
      const nbins = 100, binW = (Z_MAX - Z_MIN) / nbins;
      const counts = new Float64Array(nbins);
      for (let i = 0; i < samples.length; i++) {
        let b = Math.floor((samples[i] - Z_MIN) / binW);
        b = clip(b, 0, nbins - 1);
        counts[b] += 1;
      }
      const histDens = Array.from(counts, c => c / (samples.length * binW));

      // analytic continuous density on a fine grid
      const ng = 400;
      const grid = d3.range(ng + 1).map(i => Z_MIN + (Z_MAX - Z_MIN) * i / ng);
      const totalCurve = grid.map(xx => densCont(xx, zstar, g));
      const atoms = atomList(zstar, g);

      // y-scale: fit the continuous curve (atoms shown as capped arrows, so they
      // don't blow up the axis); keep histogram bars clamped to the same range.
      const curveMax = d3.max(totalCurve);
      const yMax = Math.max(curveMax * 1.25, g.c_rand / (Z_MAX - Z_MIN) * 3, 0.4);
      const y = d3.scaleLinear().domain([0, yMax]).range([H - M.b, M.t]);

      // axes
      gAxes.selectAll("*").remove();
      gAxes.append("g").attr("transform", `translate(0,${H - M.b})`)
        .call(d3.axisBottom(x).ticks(6).tickSize(4)).call(s => s.selectAll("text").style("font-size", "10px").style("fill", "#666"))
        .call(s => s.selectAll("line,path").style("stroke", C.axis));
      gAxes.append("g").attr("transform", `translate(${M.l},0)`)
        .call(d3.axisLeft(y).ticks(4).tickSize(4)).call(s => s.selectAll("text").style("font-size", "10px").style("fill", "#666"))
        .call(s => s.selectAll("line,path").style("stroke", C.axis));
      gAxes.append("text").attr("x", (M.l + W - M.r) / 2).attr("y", H - 4)
        .style("font-size", "11px").style("fill", "#555").style("text-anchor", "middle").text("measured range z (m)");
      gAxes.append("text").attr("transform", `translate(13,${(M.t + H - M.b) / 2}) rotate(-90)`)
        .style("font-size", "11px").style("fill", "#555").style("text-anchor", "middle").text("density");

      // histogram bars (clamped)
      gHist.selectAll("rect").data(histDens).join("rect")
        .attr("x", (d, i) => x(Z_MIN + i * binW) + 0.4)
        .attr("width", Math.max(x(binW) - x(0) - 0.8, 0.5))
        .attr("y", d => y(Math.min(d, yMax)))
        .attr("height", d => (H - M.b) - y(Math.min(d, yMax)))
        .style("fill", C.hist);

      // per-component continuous curves (weighted)
      const line = d3.line().x(d => x(d[0])).y(d => y(Math.min(d[1], yMax)));
      const comps = [];
      if (g.sigma_hit >= SIG_THR)
        comps.push({ color: C.hit, pts: grid.map(xx => [xx, g.c_hit * normPdf(xx, zstar, g.sigma_hit)]) });
      comps.push({ color: C.rand, pts: grid.map(xx => [xx, g.c_rand / (Z_MAX - Z_MIN)]) });
      if (g.c_short > 0)
        comps.push({ color: C.short, pts: grid.map(xx => [xx, g.c_short * shortPdf(xx, zstar, LAMBDA_SHORT)]) });
      gComp.selectAll("path").data(comps).join("path")
        .attr("d", d => line(d.pts)).style("fill", "none")
        .style("stroke", d => d.color).style("stroke-width", 1.4).style("stroke-dasharray", "3 2").style("opacity", 0.85);

      // total mixture density (bold)
      gTotal.selectAll("path").data([grid.map((xx, i) => [xx, totalCurve[i]])]).join("path")
        .attr("d", line).style("fill", "none").style("stroke", C.total).style("stroke-width", 2);

      // atoms: capped vertical arrow at the atom location + mass label
      gAtom.selectAll("*").remove();
      atoms.forEach(a => {
        if (a.mass < 2e-3) return;
        const px = x(a.z);
        gAtom.append("line").attr("x1", px).attr("x2", px).attr("y1", H - M.b).attr("y2", M.t + 6)
          .style("stroke", a.color).style("stroke-width", 2.2);
        gAtom.append("path").attr("d", `M${px - 4},${M.t + 11} L${px + 4},${M.t + 11} L${px},${M.t + 4} Z`)
          .style("fill", a.color);
        gAtom.append("text").attr("x", px + (a.z >= Z_MAX - 1e-6 ? -4 : 4)).attr("y", M.t + 1)
          .style("font-size", "9.5px").style("font-weight", "600").style("fill", a.color)
          .style("text-anchor", a.z >= Z_MAX - 1e-6 ? "end" : "start")
          .text("atom " + a.mass.toFixed(3));
      });

      // z* marker
      gMark.selectAll("*").remove();
      gMark.append("line").attr("x1", x(zstar)).attr("x2", x(zstar)).attr("y1", H - M.b).attr("y2", M.t)
        .style("stroke", C.zstar).style("stroke-width", 1.3).style("stroke-dasharray", "5 3").style("opacity", 0.7);
      gMark.append("text").attr("x", x(zstar)).attr("y", M.t - 6).style("font-size", "10px")
        .style("font-weight", "600").style("fill", C.zstar).style("text-anchor", "middle").text("z*");
    }

    // ── legend (above the plot; component weights for the current grade) ──
    function drawLegend() {
      const g = GRADES[grade];
      const items = [
        { c: C.total, t: "mixture p(z|z*)", w: null, solid: true },
        { c: C.hit, t: "hit", w: g.c_hit },
        { c: C.max, t: "max", w: g.c_max },
        { c: C.rand, t: "rand", w: g.c_rand },
        { c: C.short, t: "short", w: g.c_short },
      ];
      legendDiv.html("");
      items.forEach(it => {
        const cell = legendDiv.append("span").style("display", "inline-flex").style("align-items", "center").style("gap", "5px");
        cell.append("span").style("display", "inline-block").style("width", "16px").style("height", "0")
          .style("border-top", `${it.solid ? 2 : 1.5}px ${it.solid ? "solid" : "dashed"} ${it.c}`);
        cell.append("span").html(it.w === null ? it.t : `c<sub>${it.t}</sub> = ${it.w.toFixed(3)}`);
      });
    }

    // ── grade param readout + stats ──
    function drawSide() {
      const g = GRADES[grade];
      GRADE_ORDER.forEach(q => {
        const on = q === grade;
        gradeBtns[q].style("background", on ? "#333" : "#fff").style("color", on ? "#fff" : "#222")
          .style("border-color", on ? "#333" : "#bbb");
      });
      gradeInfo.html(
        `σ<sub>hit</sub> = ${g.sigma_hit}&nbsp;m &nbsp;·&nbsp; per-ray energy ×${GRADE_COST[grade].toFixed(2)}<br>` +
        `c<sub>hit</sub>=${g.c_hit} &nbsp; c<sub>max</sub>=${g.c_max} &nbsp; c<sub>rand</sub>=${g.c_rand}`
      );

      const zstar = val("zstar");
      let s = 0, s2 = 0;
      for (let i = 0; i < samples.length; i++) { s += samples[i]; s2 += samples[i] * samples[i]; }
      const mean = s / samples.length;
      const std = Math.sqrt(Math.max(s2 / samples.length - mean * mean, 0));
      const acc = Math.abs(mean - zstar);
      statsPanel.html("");
      const row = (label, value, unit) => {
        const r = statsPanel.append("div").style("display", "flex").style("font-size", "11.5px").style("align-items", "baseline");
        r.append("span").style("color", "#555").html(label);
        r.append("span").style("margin-left", "auto").style("font-weight", "600").style("color", "#222").text(value);
        r.append("span").style("color", "#999").style("font-size", "9.5px").style("margin-left", "3px").text(unit);
      };
      row("accuracy &nbsp;|E[z]−z*|", acc.toFixed(3), "m");
      row("precision &nbsp;σ[z]", std.toFixed(3), "m");
      row("E[z]", mean.toFixed(3), "m");
    }

    // ── cost panel ──
    const COST_COLOR = { quality: "#2E86AB", mem: "#E0A458" };
    function drawCost() {
      const nrays = NRAYS[val("c-nrays")];
      const history = HISTORY[val("c-hist")];
      const c = sensorCost(grade, nrays, history);
      const pct = x => (100 * x / Math.max(c.total, 1e-9)).toFixed(0);
      costPanel.html("");
      const head = costPanel.append("div").style("display", "flex").style("align-items", "baseline").style("gap", "6px");
      head.append("span").style("font-size", "20px").style("font-weight", "700").style("color", "#222").text(c.total.toFixed(1));
      head.append("span").style("font-size", "10px").style("color", "#999")
        .html(`= (energy<sub>${grade}</sub> + mem·log₂H) · n<sub>rays</sub>`);
      const bar = costPanel.append("div").style("display", "flex").style("height", "14px")
        .style("border-radius", "3px").style("overflow", "hidden").style("border", "1px solid #ddd");
      bar.append("div").style("width", pct(c.quality) + "%").style("background", COST_COLOR.quality);
      bar.append("div").style("width", pct(c.mem) + "%").style("background", COST_COLOR.mem);
      const row = (color, label, formula, value) => {
        const r = costPanel.append("div").style("display", "flex").style("align-items", "center").style("gap", "5px").style("font-size", "11px");
        r.append("span").style("width", "9px").style("height", "9px").style("border-radius", "2px").style("background", color).style("flex-shrink", "0");
        r.append("span").style("color", "#444").html(label);
        r.append("span").style("color", "#aaa").style("font-size", "9px").html(formula);
        r.append("span").style("margin-left", "auto").style("font-weight", "600").style("color", "#222").text(value.toFixed(1));
      };
      row(COST_COLOR.quality, "grade energy", `${GRADE_COST[grade].toFixed(2)}·${nrays}`, c.quality);
      row(COST_COLOR.mem, "memory", `0.5·log₂(${history})·${nrays}`, c.mem);
      costPanel.append("div").style("font-size", "9px").style("color", "#aaa").style("line-height", "1.4")
        .html("Per-ray grade energy is fit from SNR/accuracy/precision over the beam model; cost scales linearly with the ray count.");
    }

    // ── orchestration ──
    function resample() { samples = sampleMany(val("zstar"), GRADES[grade], N_SAMPLES); }
    function drawAll() { drawDist(); drawLegend(); drawSide(); drawCost(); }
    function layout() {
      const w = Math.floor(distCol.node().getBoundingClientRect().width);
      W = Math.max(w || 520, 360);
      distSvg.attr("width", W);
    }
    let rT;
    window.addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(() => { layout(); drawAll(); }, 100); });

    addSlider(ctl, { label: "z* &mdash; true distance (m)", id: "zstar", min: 0.2, max: 9.8, step: 0.1, value: 4.0, fmt: v => (+v).toFixed(1) },
      () => { resample(); drawAll(); });
    addSlider(costNrays, { label: "n<sub>rays</sub>", id: "c-nrays", min: 0, max: NRAYS.length - 1, step: 1, value: 12, fmt: i => NRAYS[+i] }, drawCost);
    addSlider(costHist, { label: "history H", id: "c-hist", min: 0, max: HISTORY.length - 1, step: 1, value: 0, fmt: i => HISTORY[+i] }, drawCost);
    resampleBtn.on("click", () => { resample(); drawAll(); });

    resample();
    layout();
    drawAll();
  }

  function tryInit() {
    if (typeof d3 !== "undefined") init();
    else setTimeout(tryInit, 50);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", tryInit);
  else tryInit();
})();
