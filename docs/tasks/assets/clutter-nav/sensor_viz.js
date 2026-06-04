// Interactive radar-sensor visualization for the clutter-nav task — uses D3 v7
// Left: the world scene (sampleable). Right: the ego-centric radar observation
// produced by scanning that scene (independently sampleable).
//
// Ports lead/tasks/nav/{task.reset, sensor.scan} to the browser.
(function () {
  const CONTAINER_ID = "clutter-nav-sensors";
  const SIZE = 360;

  // ── Task constants (appendix §clutter-nav) ────────────────────────────────
  const TASK = {
    world_size: 15.0,        // plane is [-15, 15]^2
    n_static: [5, 20],       // uniform random count per reset
    n_dynamic: [0, 15],
    agent_r: 0.5,
    obstacle_r: [0.3, 2.0],
    goal_tol: 0.25,
    goal_ego_R: [10.0, 15.0],
    exo_speed: [0.5, 4.0],   // |v| for dynamic obstacles (task caps v at 5 m/s)
  };

  // ── Fixed sensor parameters (held constant across modes; appendix §nav:sensor)
  const FIXED = {
    res_deg: 1.0,            // 1° angular bins
    sigma_r: 0.2,           // base noise scales σ̂_*
    sigma_az: 0.05,
    sigma_sz: 0.08,
    sigma_v: 0.3,
    fp_rate: 1.0,           // Poisson(1) spurious detections per scan
    p_fp: 1e-5,
    max_vel: 10.0,
    v_clim: 5.0,            // velocity colormap range (m/s)
  };

  // ── Cost model (appendix §nav:cost; weights from config/ga/1/cost.yaml) ───
  const COST = {
    w_s: 1.0, w_c: 0.5,                 // sensor-energy & coverage weights
    R0_ref: 2.0,                        // R_ref
    A_ref: 0.5 * 100 * (Math.PI / 3),   // smallest wedge: 0.5 · 10² · (π/3)
  };
  function sensorCost(cfg) {
    const C_sens = Math.pow(cfg.r0 / COST.R0_ref, 2);
    const C_cov = (0.5 * cfg.max_range * cfg.max_range * cfg.fov) / COST.A_ref;
    const sens = COST.w_s * C_sens, cov = COST.w_c * C_cov;
    return { C_sens, C_cov, sens, cov, total: sens + cov };
  }

  // ── Palette (matches the matplotlib renderers) ────────────────────────────
  const C = {
    bg: "#FAFAFA", grid: "#E5E5E5", spoke: "#E0E0E0",
    ego: "#2E86AB", goal: "#1B4965",
    staticObs: "#9AA1A8", dynObs: "#E07B5F",
    fov: "#BBBBBB", det: "#333333", fp: "#111111",
  };

  // ── RNG helpers ───────────────────────────────────────────────────────────
  const U = (a, b) => a + Math.random() * (b - a);
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function poisson(lam) {
    if (lam <= 0) return 0;
    const L = Math.exp(-lam);
    let k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
  }
  const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const wrap = a => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

  // ── Scene sampling (≈ Task.reset) ─────────────────────────────────────────
  function sampleScene() {
    const W = TASK.world_size, inner = 0.95 * W;
    const nS = Math.round(U(TASK.n_static[0], TASK.n_static[1]));
    const nD = Math.round(U(TASK.n_dynamic[0], TASK.n_dynamic[1]));
    const N = nS + nD;
    const exo = [];
    for (let i = 0; i < N; i++) {
      const isStatic = i < nS;
      const r = isStatic
        ? U(TASK.obstacle_r[0], TASK.obstacle_r[1])
        : TASK.agent_r + U(-0.1 * TASK.agent_r, 0.1 * TASK.agent_r);
      let vx = 0, vy = 0;
      if (!isStatic) {
        const sp = U(TASK.exo_speed[0], TASK.exo_speed[1]), dir = U(0, 2 * Math.PI);
        vx = sp * Math.cos(dir); vy = sp * Math.sin(dir);
      }
      exo.push({ x: U(-W, W), y: U(-W, W), r, vx, vy, isStatic });
    }

    // ego: collision-free placement, random heading
    let ego = { x: 0, y: 0, theta: U(0, 2 * Math.PI) };
    for (let t = 0; t < 400; t++) {
      const ex = U(-inner, inner), ey = U(-inner, inner);
      let ok = true;
      for (const o of exo) {
        if (Math.hypot(o.x - ex, o.y - ey) - TASK.agent_r - o.r < 0) { ok = false; break; }
      }
      if (ok) { ego = { x: ex, y: ey, theta: U(0, 2 * Math.PI) }; break; }
    }

    // ego goal: obstacle-free ball within bounds
    let goal = [ego.x, ego.y];
    for (let t = 0; t < 400; t++) {
      const rr = U(TASK.goal_ego_R[0], TASK.goal_ego_R[1]), th = U(0, 2 * Math.PI);
      const gx = ego.x + rr * Math.cos(th), gy = ego.y + rr * Math.sin(th);
      if (Math.abs(gx) > inner || Math.abs(gy) > inner) continue;
      goal = [gx, gy];
      let ok = true;
      for (const o of exo) {
        if (Math.hypot(o.x - gx, o.y - gy) - TASK.agent_r - o.r < TASK.goal_tol) { ok = false; break; }
      }
      if (ok) break;
    }
    return { exo, ego, goal, nS, nD };
  }

  // ── Radar scan (≈ sensor.scan) ────────────────────────────────────────────
  function scan(scene, cfg) {
    const { exo, ego } = scene;
    const cand = exo.map((o, i) => {
      const dx = o.x - ego.x, dy = o.y - ego.y;
      const dist = Math.hypot(dx, dy);
      const phi = wrap(Math.atan2(dy, dx) - ego.theta);
      return { o, i, dist, phi, dx, dy };
    });

    const inRange = cand.filter(c => c.dist < cfg.max_range && Math.abs(c.phi) <= cfg.fov / 2);
    inRange.sort((a, b) => a.dist - b.dist);

    // occlusion: nearest obstacle wins per angular bin
    const resRad = cfg.res_deg * Math.PI / 180;
    const seen = new Set();
    const visible = [];
    for (const c of inRange) {
      const b = Math.floor(c.phi / resRad);
      if (!seen.has(b)) { seen.add(b); visible.push(c); }
    }

    const real = [];
    for (const c of visible) {
      const P = Math.pow(cfg.r0 / c.dist, 4);
      const p_d = Math.pow(cfg.p_fp, 1 / (1 + P));
      if (Math.random() >= p_d) continue; // missed detection
      const sd = cfg.r0 / Math.sqrt(P);
      const v_r_true = (c.o.vx * c.dx + c.o.vy * c.dy) / Math.max(c.dist, 1e-6);
      real.push({
        r: clip(c.dist + cfg.sigma_r * sd * randn(), 0, cfg.max_range),
        az: wrap(c.phi + cfg.sigma_az * sd * randn()),
        sz: clip(c.o.r + cfg.sigma_sz * sd * randn(), 0.1, 3.0),
        v_r: clip(v_r_true + cfg.sigma_v * sd * randn(), -cfg.max_vel, cfg.max_vel),
        is_false: false, src: c.i,
      });
    }

    const fp = [];
    const nFp = poisson(cfg.fp_rate);
    for (let i = 0; i < nFp; i++) {
      const sz = U(0.1, 3.0);
      // place the blip so its disc clears the robot's footprint in the ego frame
      // (centers ≥ agent_r + sz apart); fall back to max_range if it can't fit
      const rMin = Math.min(TASK.agent_r + sz, cfg.max_range);
      fp.push({
        r: U(rMin, cfg.max_range),
        az: U(-cfg.fov / 2, cfg.fov / 2),
        sz,
        v_r: 2.0 * randn(),
        is_false: true, src: -1,
      });
    }
    return { real, fp, nVisible: visible.length };
  }

  // ── Velocity colormap (coolwarm; centred at 0) ────────────────────────────
  function vColor(v, vmax) {
    const t = clip((vmax - v) / (2 * vmax), 0, 1); // +v → red, -v → blue
    return d3.interpolateRdBu(t);
  }

  // ── DOM helpers ───────────────────────────────────────────────────────────
  function addSlider(parent, { label, id, min, max, step, value, fmt }, onChange) {
    const row = parent.append("div").style("display", "flex").style("flex-direction", "column")
      .style("gap", "1px").style("margin-bottom", "5px");
    const top = row.append("div").style("display", "flex").style("justify-content", "space-between")
      .style("font-size", "12px");
    top.append("span").html(label);
    const valSpan = top.append("span").style("font-weight", "600").text((fmt || (v => v))(value));
    row.append("input").attr("type", "range").attr("id", id).attr("min", min).attr("max", max)
      .attr("step", step).attr("value", value).style("width", "100%")
      .on("input", function () { valSpan.text((fmt || (v => v))(this.value)); onChange(); });
  }
  const val = id => { const el = document.getElementById(id); return el ? +el.value : 0; };

  // ── Main ──────────────────────────────────────────────────────────────────
  function init() {
    const container = d3.select("#" + CONTAINER_ID);
    if (container.empty()) return;
    container.html("");

    let scene = sampleScene();
    let dets = null;

    const outer = container.append("div")
      .style("display", "flex").style("gap", "20px").style("align-items", "flex-start").style("flex-wrap", "wrap");

    // ── Left: controls ──
    const left = outer.append("div").style("width", "210px").style("flex-shrink", "0")
      .style("display", "flex").style("flex-direction", "column").style("gap", "7px");

    const btnRow = left.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "5px");
    const btnStyle = b => b.style("font-size", "12px").style("padding", "6px 8px").style("cursor", "pointer")
      .style("border", "1px solid #bbb").style("border-radius", "5px").style("background", "#fff").style("color", "#222");
    const sceneBtn = btnStyle(btnRow.append("button")).html("⟳ Sample scene");
    const obsBtn = btnStyle(btnRow.append("button")).html("⚡ Sample observation");

    left.append("div").style("font-weight", "700").style("font-size", "13px").style("color", "#333")
      .style("margin-top", "3px").text("Radar config");
    const ctl = left.append("div");

    // ── Sensor cost panel (filled by renderCost) ──
    left.append("div").style("font-weight", "700").style("font-size", "13px").style("color", "#333")
      .style("margin-top", "5px").text("Sensor cost / step");
    const costPanel = left.append("div")
      .style("background", "#f8f8f8").style("border", "1px solid #e8e8e8").style("border-radius", "5px")
      .style("padding", "7px 9px").style("display", "flex").style("flex-direction", "column").style("gap", "5px");

    // ── Right: scene + sensor views ──
    const right = outer.append("div").style("display", "flex").style("gap", "16px")
      .style("align-items", "flex-start").style("flex", "1").style("flex-wrap", "wrap");

    const sceneCol = right.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "4px");
    sceneCol.append("div").style("font-size", "12px").style("font-weight", "700").style("color", "#333")
      .text("Scene (world frame)");
    const sceneSvg = sceneCol.append("svg").attr("width", SIZE).attr("height", SIZE)
      .style("border", "1px solid #ddd").style("border-radius", "4px").style("display", "block").style("background", C.bg);
    const sceneInfo = sceneCol.append("div").style("font-size", "10px").style("color", "#888").style("min-height", "13px");

    const sensorCol = right.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "4px");
    sensorCol.append("div").style("font-size", "12px").style("font-weight", "700").style("color", "#333")
      .html("Radar observation (ego frame, forward&nbsp;↑)");
    const sensorWrap = sensorCol.append("div").style("display", "flex").style("gap", "8px").style("align-items", "flex-start");
    const sensorSvg = sensorWrap.append("svg").attr("width", SIZE).attr("height", SIZE)
      .style("border", "1px solid #ddd").style("border-radius", "4px").style("display", "block").style("background", C.bg);
    const legend = sensorWrap.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "6px");
    const sensorInfo = sensorCol.append("div").style("font-size", "10px").style("color", "#888").style("min-height", "13px");

    // ── Scene layers ──
    const sL = {
      grid: sceneSvg.append("g"), fov: sceneSvg.append("g"), goal: sceneSvg.append("g"),
      obs: sceneSvg.append("g"), rays: sceneSvg.append("g"), ego: sceneSvg.append("g"),
    };
    // ── Sensor layers ──
    const rL = {
      grid: sensorSvg.append("g"), fov: sensorSvg.append("g"), dets: sensorSvg.append("g"), ego: sensorSvg.append("g"),
    };

    // ── Config getter (only the 3 designer-controlled knobs; rest fixed) ──
    function getCfg() {
      return Object.assign({}, FIXED, {
        max_range: val("cn-range"),
        r0: val("cn-r0"),
        fov: val("cn-fov") * Math.PI / 180,
      });
    }

    // ── Scene drawing (world frame, y-up) ──
    function drawScene(cfg) {
      const W = TASK.world_size, pad = W * 1.03;
      const scX = d3.scaleLinear().domain([-pad, pad]).range([0, SIZE]);
      const scY = d3.scaleLinear().domain([-pad, pad]).range([SIZE, 0]);
      const pm = (SIZE / 2) / pad; // px per metre

      // arena box + grid
      sL.grid.selectAll("*").remove();
      sL.grid.append("rect").attr("x", scX(-W)).attr("y", scY(W))
        .attr("width", 2 * W * pm).attr("height", 2 * W * pm)
        .style("fill", "none").style("stroke", C.grid).style("stroke-width", 1);

      // FOV wedge / range disk around ego
      sL.fov.selectAll("*").remove();
      const { ego, goal } = scene;
      const full = cfg.fov >= 2 * Math.PI - 1e-3;
      if (full) {
        sL.fov.append("circle").attr("cx", scX(ego.x)).attr("cy", scY(ego.y)).attr("r", cfg.max_range * pm)
          .style("fill", "rgba(46,134,171,0.05)").style("stroke", C.fov).style("stroke-width", 1).style("stroke-dasharray", "3 3");
      } else {
        const n = 48, a0 = ego.theta - cfg.fov / 2, a1 = ego.theta + cfg.fov / 2;
        let pts = [[scX(ego.x), scY(ego.y)]];
        for (let i = 0; i <= n; i++) {
          const a = a0 + (a1 - a0) * i / n;
          pts.push([scX(ego.x + cfg.max_range * Math.cos(a)), scY(ego.y + cfg.max_range * Math.sin(a))]);
        }
        sL.fov.append("path").attr("d", "M" + pts.map(p => p.join(",")).join("L") + "Z")
          .style("fill", "rgba(46,134,171,0.06)").style("stroke", C.fov).style("stroke-width", 1).style("stroke-dasharray", "3 3");
      }

      // obstacles
      const detSrc = new Set((dets ? dets.real : []).map(d => d.src));
      sL.obs.selectAll("circle").data(scene.exo).join("circle")
        .attr("cx", d => scX(d.x)).attr("cy", d => scY(d.y)).attr("r", d => Math.max(d.r * pm, 1.2))
        .style("fill", d => d.isStatic ? C.staticObs : C.dynObs)
        .style("fill-opacity", d => d.isStatic ? 0.9 : 0.85)
        .style("stroke", (d, i) => detSrc.has(i) ? "#1a5c1a" : "none")
        .style("stroke-width", (d, i) => detSrc.has(i) ? 2 : 0);

      // velocity arrows for dynamic obstacles
      sL.obs.selectAll("line.vel").data(scene.exo.filter(d => !d.isStatic)).join("line")
        .attr("class", "vel")
        .attr("x1", d => scX(d.x)).attr("y1", d => scY(d.y))
        .attr("x2", d => scX(d.x + d.vx * 0.5)).attr("y2", d => scY(d.y + d.vy * 0.5))
        .style("stroke", C.dynObs).style("stroke-width", 1.3).style("opacity", 0.8);

      // rays to detected obstacles
      sL.rays.selectAll("*").remove();
      (dets ? dets.real : []).forEach(d => {
        const o = scene.exo[d.src];
        sL.rays.append("line").attr("x1", scX(ego.x)).attr("y1", scY(ego.y))
          .attr("x2", scX(o.x)).attr("y2", scY(o.y))
          .style("stroke", "rgba(26,92,26,0.35)").style("stroke-width", 0.8);
      });

      // goal
      sL.goal.selectAll("*").remove();
      sL.goal.append("line").attr("x1", scX(ego.x)).attr("y1", scY(ego.y))
        .attr("x2", scX(goal[0])).attr("y2", scY(goal[1]))
        .style("stroke", C.goal).style("stroke-width", 0.8).style("stroke-dasharray", "4 3").style("opacity", 0.5);
      sL.goal.append("circle").attr("cx", scX(goal[0])).attr("cy", scY(goal[1])).attr("r", TASK.agent_r * pm)
        .style("fill", "none").style("stroke", C.goal).style("stroke-width", 2).style("stroke-dasharray", "3 2");

      // ego marker + heading
      sL.ego.selectAll("*").remove();
      sL.ego.append("circle").attr("cx", scX(ego.x)).attr("cy", scY(ego.y)).attr("r", Math.max(TASK.agent_r * pm, 3))
        .style("fill", C.ego);
      sL.ego.append("line").attr("x1", scX(ego.x)).attr("y1", scY(ego.y))
        .attr("x2", scX(ego.x + 2.0 * Math.cos(ego.theta))).attr("y2", scY(ego.y + 2.0 * Math.sin(ego.theta)))
        .style("stroke", C.ego).style("stroke-width", 2.2);
    }

    // ── Sensor drawing (ego frame, forward up, left-positive azimuth) ──
    function drawSensor(cfg) {
      const cx = SIZE / 2, cy = SIZE / 2;
      const pad = cfg.max_range * 1.06;
      const pm = (SIZE / 2) / pad; // px per metre
      // forward (az=0) → up; +az (CCW/left) → left
      const sx = (r, az) => cx - r * pm * Math.sin(az);
      const sy = (r, az) => cy - r * pm * Math.cos(az);

      // range rings + azimuth spokes
      rL.grid.selectAll("*").remove();
      const nRings = 5;
      for (let i = 1; i <= nRings; i++) {
        const rr = cfg.max_range * i / nRings;
        rL.grid.append("circle").attr("cx", cx).attr("cy", cy).attr("r", rr * pm)
          .style("fill", "none").style("stroke", C.grid).style("stroke-width", 0.7);
        rL.grid.append("text").attr("x", cx + 2).attr("y", cy - rr * pm - 1)
          .style("font-size", "8px").style("fill", "#aaa").text(rr.toFixed(0) + "m");
      }
      for (let a = 0; a < 360; a += 30) {
        const ar = a * Math.PI / 180;
        rL.grid.append("line").attr("x1", cx).attr("y1", cy).attr("x2", sx(cfg.max_range, ar)).attr("y2", sy(cfg.max_range, ar))
          .style("stroke", C.spoke).style("stroke-width", 0.5);
      }

      // FOV wedge
      rL.fov.selectAll("*").remove();
      if (cfg.fov < 2 * Math.PI - 1e-3) {
        const n = 48; let pts = [[cx, cy]];
        for (let i = 0; i <= n; i++) {
          const az = -cfg.fov / 2 + cfg.fov * i / n;
          pts.push([sx(cfg.max_range, az), sy(cfg.max_range, az)]);
        }
        rL.fov.append("path").attr("d", "M" + pts.map(p => p.join(",")).join("L") + "Z")
          .style("fill", "none").style("stroke", C.fov).style("stroke-width", 1.2);
      }

      // detections
      rL.dets.selectAll("*").remove();
      const all = dets ? [...dets.real, ...dets.fp] : [];
      all.forEach(d => {
        const px = sx(d.r, d.az), py = sy(d.r, d.az);
        rL.dets.append("circle").attr("cx", px).attr("cy", py).attr("r", Math.max(d.sz * pm, 2))
          .style("fill", vColor(d.v_r, cfg.v_clim)).style("fill-opacity", 0.85)
          .style("stroke", d.is_false ? C.fp : C.det)
          .style("stroke-width", d.is_false ? 1.3 : 0.6)
          .style("stroke-dasharray", d.is_false ? "2 2" : null);
      });

      // ego at centre, forward up
      rL.ego.selectAll("*").remove();
      rL.ego.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 3.5).style("fill", C.ego);
      rL.ego.append("line").attr("x1", cx).attr("y1", cy).attr("x2", cx).attr("y2", cy - 14)
        .style("stroke", C.ego).style("stroke-width", 2);
    }

    // ── Velocity legend (scaled to the ego-frame plot) ──
    function drawLegend(cfg) {
      legend.html("");
      const top = 14, h = SIZE - 70, w = 20;          // bar ≈ full plot height
      legend.append("div").style("font-size", "12px").style("font-weight", "600").style("color", "#444")
        .style("text-align", "center").html("v<sub>radial</sub><br><span style='font-size:9px;font-weight:400;color:#999'>(m/s)</span>");
      const lsvg = legend.append("svg").attr("width", 60).attr("height", h + top + 8);
      const defs = lsvg.append("defs");
      const grad = defs.append("linearGradient").attr("id", "cn-vgrad").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 1);
      for (let i = 0; i <= 10; i++) {
        const v = cfg.v_clim - 2 * cfg.v_clim * i / 10; // top=+vmax, bottom=-vmax
        grad.append("stop").attr("offset", (i * 10) + "%").attr("stop-color", vColor(v, cfg.v_clim));
      }
      lsvg.append("rect").attr("x", 6).attr("y", top).attr("width", w).attr("height", h)
        .style("fill", "url(#cn-vgrad)").style("stroke", "#bbb").style("stroke-width", 0.8);
      // tick marks + labels at +V, +V/2, 0, -V/2, -V
      const nTicks = 4;
      for (let i = 0; i <= nTicks; i++) {
        const y = top + h * i / nTicks;
        const v = cfg.v_clim - 2 * cfg.v_clim * i / nTicks;
        lsvg.append("line").attr("x1", 6 + w).attr("x2", 6 + w + 4).attr("y1", y).attr("y2", y)
          .style("stroke", "#888").style("stroke-width", 0.8);
        lsvg.append("text").attr("x", 6 + w + 7).attr("y", y + 4)
          .style("font-size", "11px").style("fill", "#555")
          .text((v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v));
      }
      legend.append("div").style("font-size", "10px").style("color", "#888").style("line-height", "1.5").style("max-width", "80px")
        .html("● size = obstacle size<br><span style='color:#111'>⬚</span> dashed = false positive");
    }

    // ── Sensor cost panel ──
    const COST_COLOR = { sens: "#2E86AB", cov: "#E0A458" };
    function renderCost(cfg) {
      const c = sensorCost(cfg);
      const pct = x => (100 * x / Math.max(c.total, 1e-9)).toFixed(0);
      costPanel.html("");
      // total
      const head = costPanel.append("div").style("display", "flex").style("align-items", "baseline").style("gap", "5px");
      head.append("span").style("font-size", "20px").style("font-weight", "700").style("color", "#222").text(c.total.toFixed(2));
      head.append("span").style("font-size", "10px").style("color", "#999").text("= w·(energy + area)");
      // stacked proportion bar
      const bar = costPanel.append("div").style("display", "flex").style("height", "14px")
        .style("border-radius", "3px").style("overflow", "hidden").style("border", "1px solid #ddd");
      bar.append("div").style("width", pct(c.sens) + "%").style("background", COST_COLOR.sens);
      bar.append("div").style("width", pct(c.cov) + "%").style("background", COST_COLOR.cov);
      // breakdown rows
      const row = (color, label, formula, value) => {
        const r = costPanel.append("div").style("display", "flex").style("align-items", "center").style("gap", "5px").style("font-size", "11px");
        r.append("span").style("width", "9px").style("height", "9px").style("border-radius", "2px").style("background", color).style("flex-shrink", "0");
        r.append("span").style("color", "#444").html(label);
        r.append("span").style("color", "#aaa").style("font-size", "9px").html(formula);
        r.append("span").style("margin-left", "auto").style("font-weight", "600").style("color", "#222").text(value.toFixed(2));
      };
      row(COST_COLOR.sens, "energy", "w<sub>s</sub>(R₀/2)²", c.sens);
      row(COST_COLOR.cov, "coverage", "w<sub>c</sub>·½·mr²·fov", c.cov);
      costPanel.append("div").style("font-size", "9px").style("color", "#aaa").style("line-height", "1.4")
        .html("Full per-step cost adds the planner's compute term and amortizes by 1/period (fixed here).");
    }

    // ── Orchestration ──
    function rescan() { dets = scan(scene, getCfg()); }
    function drawAll() {
      const cfg = getCfg();
      drawScene(cfg); drawSensor(cfg); drawLegend(cfg); renderCost(cfg);
      const nR = dets.real.length, nF = dets.fp.length;
      sceneInfo.text(`ego @ (${scene.ego.x.toFixed(1)}, ${scene.ego.y.toFixed(1)})  θ=${(scene.ego.theta * 180 / Math.PI).toFixed(0)}°  ·  ${scene.nS} static, ${scene.nD} dynamic`);
      sensorInfo.text(`detections: ${nR + nF}  (true: ${nR}, false: ${nF})  ·  in-view obstacles: ${dets.nVisible}`);
    }

    sceneBtn.on("click", () => { scene = sampleScene(); rescan(); drawAll(); });
    obsBtn.on("click", () => { rescan(); drawAll(); });

    // ── Build controls (only the designer-controlled knobs) ──
    const onKnob = () => { rescan(); drawAll(); };
    addSlider(ctl, { label: "R₀ &mdash; sensor power (m)", id: "cn-r0", min: 2, max: 50, step: 1, value: 20, fmt: v => (+v).toFixed(0) }, onKnob);
    addSlider(ctl, { label: "max range (m)", id: "cn-range", min: 5, max: 15, step: 0.5, value: 10, fmt: v => (+v).toFixed(1) }, onKnob);
    addSlider(ctl, { label: "FOV (deg)", id: "cn-fov", min: 60, max: 360, step: 5, value: 360, fmt: v => (+v).toFixed(0) }, onKnob);

    ctl.append("div").style("font-size", "10px").style("color", "#999").style("line-height", "1.4").style("margin-top", "6px")
      .html("Noise (σ̂<sub>r</sub>,σ̂<sub>az</sub>,σ̂<sub>sz</sub>,σ̂<sub>v</sub>), 1° binning, and Poisson(1) clutter are fixed across modes.");

    rescan();
    drawAll();
  }

  function tryInit() {
    if (typeof d3 !== "undefined") init();
    else setTimeout(tryInit, 50);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", tryInit);
  else tryInit();
})();
