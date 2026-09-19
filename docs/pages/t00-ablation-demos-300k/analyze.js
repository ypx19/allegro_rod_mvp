/* noscroll — slider updates still + numbers in place; no page scroll, no per-step table */
const ASSET_V = "noscroll";
const ORDER = ["A", "B", "C", "D"];
const RAD2DEG = 180 / Math.PI;
const DT_S = 0.04;

let DATA = null;
let current = "A";
let steps = [];
let meta = {};
let events = {};
let playing = false;
let timer = null;
const lastIdx = { A: null, B: null, C: null, D: null };
const scr = document.getElementById("scr");

function fmt(x, n = 2) {
  if (x == null || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(n);
}

function enrichPack(pack) {
  const dt = Number(pack.meta.dt_s) || DT_S;
  let prev = null;
  for (const s of pack.steps) {
    s.axial_omega_deg_s = Number(s.axial_omega_rad_s || 0) * RAD2DEG;
    s.lateral_omega_deg_s = Number(s.lateral_omega_rad_s || 0) * RAD2DEG;
    s.contact_force_total_n = (s.fingers || []).reduce((a, f) => a + Number(f.force_n || 0), 0);
    if (s.tilt_vel_rad_s == null || s.tilt_vel_deg_s == null) {
      if (prev == null) {
        s.tilt_vel_rad_s = 0;
        s.tilt_vel_deg_s = 0;
      } else {
        s.tilt_vel_rad_s = (Number(s.axis_tilt_rad) - Number(prev.axis_tilt_rad)) / dt;
        s.tilt_vel_deg_s = (Number(s.axis_tilt_deg) - Number(prev.axis_tilt_deg)) / dt;
      }
    }
    prev = s;
  }
  pack.meta.tilt_vel_source =
    pack.meta.tilt_vel_source ||
    "finite difference of tilt angle / dt (dt=0.04 s at 25 Hz); not ω_perp";
  pack.meta.axial_omega_source =
    pack.meta.axial_omega_source || "ω_axial = −ω·â, same sign as unwrap";
  pack.meta.omega_perp_source =
    pack.meta.omega_perp_source || "ω_perp = ||ω − (ω·â)â|| lateral angular rate";
}

function loadCondition(name, jump) {
  current = name;
  const pack = DATA[name];
  steps = pack.steps;
  meta = pack.meta;
  events = pack.events;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.dataset.cond === name));
  const clip = document.getElementById("cliplink");
  if (clip) {
    clip.href = meta.video || "#";
    clip.textContent = "Open MP4 clip · " + name;
  }
  const keys = [
    ["support-loss", events.first_support_loss_step],
    ["n = 0", events.first_n0_step],
    ["tilt-kill", events.first_tilt_kill_step],
  ];
  document.getElementById("keythumbs").innerHTML = keys
    .map(([lab, st]) => {
      if (st == null) return "";
      return `<button type="button" data-step="${st}">${lab} · step ${st}</button>`;
    })
    .join("");
  document.getElementById("keythumbs").querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => setIdx(Number(btn.dataset.step));
  });
  scr.max = steps.length - 1;
  const start =
    jump != null
      ? jump
      : lastIdx[name] != null
        ? lastIdx[name]
        : 0;
  setIdx(start);
}

function drawChart(idx) {
  const cv = document.getElementById("chart");
  const ctx = cv.getContext("2d");
  const w = cv.width;
  const h = cv.height;
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 52, r: 10, t: 6, b: 16 };
  const nBand = 4;
  const gap = 8;
  const usable = h - pad.t - pad.b - gap * (nBand - 1);
  const bh = usable / nBand;
  const n = Math.max(1, steps.length - 1);
  const x = (i) => pad.l + (i / n) * (w - pad.l - pad.r);
  const x0 = x(events.first_support_loss_step ?? 0);
  const x1 = x(events.first_tilt_kill_step ?? n);

  const bands = [
    {
      label: "n_contact",
      color: "#5cb8a0",
      vals: steps.map((s) => s.contact_count),
      ymin: 0,
      ymax: 3,
      bars: true,
    },
    {
      label: "tilt °",
      color: "#e8a54b",
      vals: steps.map((s) => s.axis_tilt_deg),
      ymin: 0,
      ymax: Math.max(20, ...steps.map((s) => s.axis_tilt_deg)),
    },
    {
      label: "ω_axial rad/s",
      color: "#4c8dde",
      vals: steps.map((s) => s.axial_omega_rad_s),
      signed: true,
    },
    {
      label: "dθ/dt °/s",
      color: "#c45c48",
      vals: steps.map((s) => s.tilt_vel_deg_s),
      signed: true,
    },
  ];

  bands.forEach((b, bi) => {
    const top = pad.t + bi * (bh + gap);
    let ymin = b.ymin;
    let ymax = b.ymax;
    if (b.signed) {
      const m = Math.max(0.5, ...b.vals.map(Math.abs));
      ymin = -m;
      ymax = m;
    }
    const y = (v) => top + (1 - (v - ymin) / (ymax - ymin || 1)) * bh;
    ctx.fillStyle = "rgba(255,255,255,0.025)";
    ctx.fillRect(pad.l, top, w - pad.l - pad.r, bh);
    ctx.fillStyle = "rgba(232,165,75,0.10)";
    ctx.fillRect(Math.min(x0, x1), top, Math.abs(x1 - x0), bh);
    if (b.signed || (!b.bars && ymin <= 0 && ymax >= 0)) {
      ctx.strokeStyle = "#243040";
      ctx.beginPath();
      ctx.moveTo(pad.l, y(0));
      ctx.lineTo(w - pad.r, y(0));
      ctx.stroke();
    }
    if (b.bars) {
      ctx.fillStyle = b.color;
      b.vals.forEach((v, i) => {
        const y0 = y(0);
        const y1 = y(v);
        ctx.fillRect(x(i) - 2, Math.min(y0, y1), 4, Math.max(1, Math.abs(y1 - y0)));
      });
    } else {
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      b.vals.forEach((v, i) => {
        ctx[i ? "lineTo" : "moveTo"](x(i), y(v));
      });
      ctx.stroke();
      ctx.lineWidth = 1;
    }
    ctx.fillStyle = "#8b9aab";
    ctx.font = "11px 'Sora', ui-sans-serif";
    ctx.fillText(b.label, 6, top + 12);
  });

  ctx.strokeStyle = "#e8eef4";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x(idx), pad.t);
  ctx.lineTo(x(idx), h - pad.b);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#e8eef4";
  ctx.beginPath();
  ctx.arc(x(idx), pad.t + 3, 3, 0, 6.28);
  ctx.fill();
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function render(idx) {
  const s = steps[idx];
  const frameEl = document.getElementById("frame");
  if (frameEl.getAttribute("src") !== s.frame) {
    frameEl.src = s.frame;
  }
  setText(
    "tlabel",
    `${current}   t = ${fmt(s.t_s, 2)} s   ·   step ${s.step} / ${steps[steps.length - 1].step}   ·   25 Hz`
  );
  setText(
    "winlabel",
    `loss ${events.first_support_loss_step} → n0 ${events.first_n0_step} → kill ${events.first_tilt_kill_step}`
  );
  setText("v_tilt", fmt(s.axis_tilt_deg) + "°");
  setText("v_tilt_u", fmt(s.axis_tilt_rad, 4) + " rad");
  setText("v_dtilt", fmt(s.tilt_vel_deg_s, 1) + " °/s");
  setText("v_dtilt_u", fmt(s.tilt_vel_rad_s, 3) + " rad/s  ·  Δθ/dt");
  setText("v_c", s.contact_count + " / 3");
  setText("v_c_u", "F > 0.05 N");
  setText("v_w", fmt(s.axial_omega_rad_s) + " rad/s");
  setText("v_w_u", fmt(s.axial_omega_deg_s, 1) + " °/s  ·  −ω·â");
  setText("v_lat", fmt(s.lateral_omega_rad_s) + " rad/s");
  setText("v_lat_u", fmt(s.lateral_omega_deg_s, 1) + " °/s  ·  ≠ dθ/dt");
  setText("v_ftot", fmt(s.contact_force_total_n, 2) + " N");
  setText("v_ftot_u", "Σ |mj_contactForce[0]|");
  setText("v_rot", fmt(s.axis_rotation_deg, 1) + "°");
  setText("v_rew", s.reward == null ? "—" : fmt(s.reward, 2));
  setText("v_grot", fmt(s.reward_rotation, 2));
  setText("v_wob", fmt(s.reward_low_support_wobble, 2));
  document.getElementById("k_tilt").className = "kpi" + (s.axis_tilt_deg > 14 ? " bad" : "");
  document.getElementById("k_dtilt").className = "kpi" + (s.tilt_vel_deg_s > 30 ? " bad" : "");
  document.getElementById("k_c").className = "kpi" + (s.contact_count < 2 ? " bad" : " ok");
  document.getElementById("k_ftot").className = "kpi" + (s.contact_force_total_n < 0.15 ? " bad" : "");
  const tipMm = (s.tip_error_m * 1000).toFixed(2) + " mm";
  setText("v_tip", tipMm);
  setText("v_tip_tbl", tipMm);
  setText("v_ncon", String(s.ncon));
  setText("v_axis", s.rod_axis_world.map((v) => Number(v).toFixed(3)).join(", "));
  setText("v_term", s.termination_reason);
  setText("v_kill", s.would_eval_axis_tilt_kill ? "yes (θ>1.2 rad)" : "no");
  setText("v_rot0", fmt(s.reward_rotation_before_support_gate, 2));
  const gatedNote = meta.support_aware ? "C/D support-aware" : "A/B baseline (no wobble term)";
  setText("v_gated_note", gatedNote);
  document.getElementById("fingers").innerHTML =
    s.fingers
      .map((f) => {
        const w = Math.min(100, f.force_n / 20 * 100);
        return `<div><div class="fname">${f.name} · ${f.in_contact ? "CONTACT" : "off"} · ${f.force_n.toFixed(3)} N · dist ${(f.axis_dist_m * 1000).toFixed(1)} mm</div>
      <div class="bar"><i style="width:${w}%"></i></div></div>`;
      })
      .join("") +
    `<div class="ftot"><span class="fname">total contact force</span><span class="v">${s.contact_force_total_n.toFixed(3)} N</span></div>`;
  document.querySelectorAll("#keythumbs button").forEach((btn) => {
    btn.classList.toggle("on", Number(btn.dataset.step) === s.step);
  });
  drawChart(idx);
}

function setIdx(i) {
  i = Math.max(0, Math.min(steps.length - 1, i));
  lastIdx[current] = i;
  scr.value = i;
  render(i);
}

function buildChrome() {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = ORDER.map((name) => {
    const p = DATA[name];
    return `<button class="tab ${name.toLowerCase()}" data-cond="${name}" type="button">${name} · ${p.meta.title.replace(/\s+/g, " ")}</button>`;
  }).join("");
  tabs.querySelectorAll(".tab").forEach((btn) => {
    btn.onclick = () => loadCondition(btn.dataset.cond);
  });
  const rows = ORDER.map((name) => {
    const p = DATA[name];
    const e = p.events;
    const m = p.meta;
    return `<tr>
      <td>${name}</td>
      <td>${m.obs_dim}-D hist${m.hist}${m.support_aware ? " · support-aware" : " · baseline"}</td>
      <td class="num">${m.n_steps}</td>
      <td class="num">${m.episode_return >= 0 ? "+" : ""}${m.episode_return.toFixed(1)}</td>
      <td class="num">${m.final_rotation_deg.toFixed(0)}°</td>
      <td class="num">${m.final_tilt_deg.toFixed(0)}°</td>
      <td class="num">${e.first_support_loss_step}</td>
      <td class="num">${e.first_n0_step}</td>
      <td class="num">${e.first_tilt_kill_step}</td>
      <td class="num">${e.steps_to_tilt_after_loss}</td>
      <td>${e.recontact_after_first_loss ? "yes" : "no"}</td>
      <td>${m.termination_reason}</td>
    </tr>`;
  }).join("");
  document.getElementById("cmp-body").innerHTML = rows;
  document.getElementById("key-grid").innerHTML = ORDER.map((name) => {
    const p = DATA[name];
    const e = p.events;
    const keys = [
      ["2→1", e.first_support_loss_step],
      ["n=0", e.first_n0_step],
      ["kill", e.first_tilt_kill_step],
    ];
    return `<article class="cell ${name.toLowerCase()}">
      <div class="id">${name}</div>
      <h3>${p.meta.title}</h3>
      <p>loss ${e.first_support_loss_step} · n0 ${e.first_n0_step} · kill ${e.first_tilt_kill_step} · no recontact</p>
      <div class="keyjumps">${keys
        .map(([lab, st]) => {
          if (st == null) return "";
          return `<button type="button" data-cond="${name}" data-step="${st}">${lab} · ${st}</button>`;
        })
        .join("")}</div>
    </article>`;
  }).join("");
  document.getElementById("key-grid").querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => loadCondition(btn.dataset.cond, Number(btn.dataset.step));
  });
}

scr.addEventListener("input", () => setIdx(+scr.value));
document.getElementById("prev").onclick = () => setIdx(+scr.value - 1);
document.getElementById("next").onclick = () => setIdx(+scr.value + 1);
document.getElementById("jumpLoss").onclick = () => setIdx(events.first_support_loss_step || 0);
document.getElementById("jumpZero").onclick = () => setIdx(events.first_n0_step || 0);
document.getElementById("jumpKill").onclick = () => setIdx(events.first_tilt_kill_step || steps.length - 1);
document.getElementById("play").onclick = function () {
  playing = !playing;
  this.textContent = playing ? "Pause" : "Play 25 Hz";
  if (timer) clearInterval(timer);
  if (playing)
    timer = setInterval(() => {
      let n = +scr.value + 1;
      if (n >= steps.length) n = 0;
      setIdx(n);
    }, 40);
};
document.getElementById("chart").addEventListener("click", (e) => {
  const cv = e.currentTarget;
  const rect = cv.getBoundingClientRect();
  const scaleX = cv.width / rect.width;
  const padL = 52;
  const padR = 10;
  const px = (e.clientX - rect.left) * scaleX;
  const frac = (px - padL) / (cv.width - padL - padR);
  setIdx(Math.round(frac * (steps.length - 1)));
});
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    setIdx(+scr.value - 1);
  }
  if (e.key === "ArrowRight") {
    e.preventDefault();
    setIdx(+scr.value + 1);
  }
  if (e.key === " ") {
    e.preventDefault();
    document.getElementById("play").click();
  }
  if (e.key === "1") loadCondition("A");
  if (e.key === "2") loadCondition("B");
  if (e.key === "3") loadCondition("C");
  if (e.key === "4") loadCondition("D");
});

fetch("timelines.json?v=" + ASSET_V)
  .then((r) => {
    if (!r.ok) throw new Error("timelines.json " + r.status);
    return r.json();
  })
  .then((json) => {
    DATA = json;
    ORDER.forEach((name) => enrichPack(DATA[name]));
    buildChrome();
    loadCondition("A");
  })
  .catch((err) => {
    setText("tlabel", "Failed to load timelines.json: " + err);
  });
