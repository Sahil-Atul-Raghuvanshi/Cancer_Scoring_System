/* The six-slide comparison screen.

   Its own file rather than more of app.js, because it shares nothing with the upload
   screen except the helpers. Loaded after app.js, so the top-level `const`s there -
   `$`, `$$`, `api`, `pct`, `bars`, `state` - are already in the shared global lexical
   scope and are used rather than duplicated.

   One rule, the same one app.js follows: every number here is computed on the server and
   printed verbatim. The only judgement this file makes is which figure to tint as a
   warning, and that reads a threshold rather than re-deriving a verdict - a UI that works
   out for itself whether the comparison "passed" would be a second definition of passing,
   and the two would eventually disagree. */

const CM_ORDER = ["non_epithelium", "non_invasive_epithelium", "invasive_epithelium"];
const CM_SHORT = {
  non_epithelium: "other",
  non_invasive_epithelium: "in-situ",
  invasive_epithelium: "invasive",
};

//: Below this the row's agreement figure is tinted. Not a pass mark - there is no pass
//: mark here, because the teacher is not ground truth - just the point below which a
//: reader should go and look at the pictures rather than trust the number.
const LOOK_CLOSER_BELOW = 0.6;

async function loadSixSlides() {
  let data;
  try {
    data = await api("/sixslides");
  } catch (error) {
    $("#ss-status").textContent = "Could not list the slides: " + error.message;
    return;
  }
  state.sixslides.items = data.items;

  const select = $("#ss-model");
  if (!select.options.length && data.models.length) {
    select.innerHTML = data.models
      .map((m) => `<option value="${m}"${m === data.default_model ? " selected" : ""}>${m}</option>`)
      .join("");
  }

  const done = data.items.filter((i) => i.result).length;
  const missing = data.items.filter((i) => i.available && !i.result).length;
  const absent = data.items.filter((i) => !i.available).length;

  $("#ss-status").innerHTML =
    `<b>${done}</b> of ${data.items.length} slides compared` +
    (missing ? `, ${missing} still to do` : "") +
    (absent ? `, <span style="color:var(--bad)">${absent} slide file(s) missing</span>` : "") +
    `. Each row is a <b>${data.region_um} µm</b> window at 0.5 µm/px` +
    (done ? "." : " — nothing computed yet, press Compare.");
  $("#ss-run-missing").disabled = missing === 0;

  renderSixSummary(data.summary, done);
  renderSixMatrix(data.items);
}

function renderSixSummary(summary, done) {
  const box = $("#ss-summary");
  if (!done || !summary.tiles) {
    box.innerHTML = "";
    return;
  }

  box.innerHTML = `
    <div class="card-box">
      <h3>Across ${summary.cases} slide${summary.cases === 1 ? "" : "s"}</h3>
      <div class="bigstat">
        <div><b>${pct(summary.tile_agreement)}</b><span>the two models agree</span></div>
        <div><b style="color:var(--warn)">${summary.teacher_in_situ_student_invasive}</b>
             <span>BEETLE says in-situ,<br>ours says invasive</span></div>
        <div><b style="color:var(--warn)">${summary.teacher_invasive_student_in_situ}</b>
             <span>BEETLE says invasive,<br>ours says in-situ</span></div>
        <div><b>${summary.tiles}</b><span>tiles compared</span></div>
      </div>
      <p class="note" style="margin:.7rem 0 0">
        The two middle numbers are the ones that matter — that pair is the boundary the
        whole model exists to draw. Agreement on plain tissue is easy and is not evidence
        of much. The headline is weighted by tiles, not by slide, so a small window cannot
        outvote a large one.
      </p>
    </div>`;
}

function confusionTable(agreement) {
  const head = CM_ORDER.map((c) => `<th>${CM_SHORT[c]}</th>`).join("");
  const rows = agreement.confusion
    .map((row, i) => {
      const cells = row
        .map((v, j) => `<td class="${i === j ? "diag" : v > 0 ? "bad" : ""}">${v}</td>`)
        .join("");
      return `<tr><th style="text-align:left">${CM_SHORT[CM_ORDER[i]]}</th>${cells}</tr>`;
    })
    .join("");
  return `<table class="cm">
    <tr><th style="text-align:left">BEETLE &darr; / ours &rarr;</th>${head}</tr>
    ${rows}
  </table>`;
}

function matrixRow(item) {
  const label = item.case.replace("CAN_", "").replace("_26", "");

  if (!item.available) {
    return `<div class="matrix-row">
      <div class="who"><b>${label}</b><span style="color:var(--bad)">slide file missing</span></div>
      <div class="cell pending" style="grid-column: 2 / -1">${item.error ?? "not found"}</div>
    </div>`;
  }
  if (!item.result) {
    return `<div class="matrix-row">
      <div class="who"><b>${label}</b><span>${item.mm ? item.mm.join(" &times; ") + " mm" : ""}</span></div>
      <div class="cell pending">not compared yet</div>
      <div class="cell pending">&mdash;</div>
      <div class="cell pending">&mdash;</div>
    </div>`;
  }

  const r = item.result;
  const a = r.agreement;
  const url = (which) => `/api/sixslides/${item.case}/image/${which}`;
  // Keyed to the run, so re-comparing a slide does not show the browser's cached picture
  // of the previous one beside the new numbers.
  const v = encodeURIComponent(`${r.seconds}_${a.tile_agreement ?? 0}`);
  const iou = (name) => a.per_class[name].iou ?? "&mdash;";

  return `<div class="matrix-row">
    <div class="who">
      <b>${label}</b>
      <span>${r.grid[0]}&times;${r.grid[1]} tiles<br>${r.region_um} µm<br>${r.seconds}s</span>
    </div>

    <div class="cell">
      <img loading="lazy" src="${url("original")}?v=${v}" alt="the tissue">
      <span class="tag">the tissue</span>
    </div>

    <div class="cell blocky">
      <img loading="lazy" src="${url("ours")}?v=${v}" alt="our model">
      <span class="tag">our model &middot; 224 px tiles</span>
    </div>

    <div class="cell">
      <img loading="lazy" src="${url("beetle")}?v=${v}" alt="BEETLE">
      <span class="tag">BEETLE &middot; per pixel</span>
    </div>

    <div class="who"></div>
    <div class="agree" style="grid-column: 2 / -1; padding-top:.4rem">
      <div class="cols" style="grid-template-columns: minmax(0,1fr) minmax(0,1fr) minmax(0,1.15fr); gap:1rem">
        <div>
          <b class="${a.tile_agreement < LOOK_CLOSER_BELOW ? "off" : ""}">${pct(a.tile_agreement)}</b>
          <span class="note" style="display:block;margin:0">
            of ${a.tiles} tiles agree<br>${pct(a.pixel_agreement)} per pixel
          </span>
        </div>
        <div>
          <span class="note" style="display:block;margin:0">
            <span class="chip c1"></span>in-situ IoU ${iou("non_invasive_epithelium")}<br>
            <span class="chip c2"></span>invasive IoU ${iou("invasive_epithelium")}<br>
            <span class="chip c0"></span>other IoU ${iou("non_epithelium")}
          </span>
        </div>
        <div>${confusionTable(a)}</div>
      </div>

      <details style="margin-top:.4rem">
        <summary>Where exactly they differ, and which window was chosen</summary>
        <div class="detail-body">
          <div class="cell blocky" style="max-width:520px;margin:.3rem 0 .7rem">
            <img loading="lazy" src="${url("disagreement")}?v=${v}" alt="disagreement">
            <span class="tag">magenta = the two models chose different classes</span>
          </div>
          <table class="kv" style="max-width:540px">
            <tr><td>window on the slide</td><td>x ${r.window.x}, y ${r.window.y}</td></tr>
            <tr><td>tissue in the window</td><td>${pct(r.window.tissue_share)}</td></tr>
            <tr><td>nuclear density it was picked by</td><td>${r.window.nuclear_density}</td></tr>
            <tr><td>windows scored on this slide</td><td>${r.window.candidates_scored}</td></tr>
            <tr><td>resolution</td><td>${r.slide_mpp} &rarr; ${r.working_mpp} µm/px</td></tr>
            <tr><td>our model</td><td><code>${r.student.model}</code></td></tr>
            <tr><td>its confidence</td><td>${pct(r.student.mean_confidence)}</td></tr>
            <tr><td>BEETLE folds &middot; confidence</td>
                <td>${r.teacher.folds.length} of 5 &middot; ${pct(r.teacher.mean_confidence)}</td></tr>
          </table>
          <p class="note" style="margin:.7rem 0 .2rem">How much of the window each called what:</p>
          <div class="cols" style="grid-template-columns:1fr 1fr;gap:1rem">
            <div><p class="note" style="margin:.2rem 0">our model</p>${bars(r.student.class_area_fraction)}</div>
            <div><p class="note" style="margin:.2rem 0">BEETLE</p>${bars(r.teacher.class_area_fraction)}</div>
          </div>
        </div>
      </details>
    </div>
  </div>`;
}

function renderSixMatrix(items) {
  $("#ss-matrix").innerHTML = `<div class="matrix">
    <div class="matrix-head">
      <span>patient</span>
      <span><b>the tissue</b></span>
      <span><b>our model</b> &mdash; trained here</span>
      <span><b>BEETLE</b> &mdash; it taught ours</span>
    </div>
    ${items.map(matrixRow).join("")}
  </div>`;
}

async function runSixSlides(cases) {
  try {
    await api("/sixslides/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cases: cases ?? [],
        folds: $("#ss-folds").value.split(",").map(Number),
        model_name: $("#ss-model").value || null,
      }),
    });
  } catch (error) {
    alert("Could not start: " + error.message);
    return;
  }
  pollSixSlides();
}

async function pollSixSlides() {
  if (state.sixslides.polling) clearInterval(state.sixslides.polling);

  const tick = async () => {
    let data;
    try {
      data = await api("/jobs?limit=8");
    } catch {
      return;
    }
    // Only this screen's jobs. The upload screen queues its own into the same runner,
    // and showing those here would report an unrelated image's progress as this one's.
    const mine = data.items.filter((job) => job.kind === "sixslides");

    $("#ss-jobs").innerHTML = mine
      .slice(0, 2)
      .map((job) => {
        const s = job.result?.summary;
        return `<div class="job ${job.status}">
          <div class="job-head">
            <b>${job.label}</b>
            <span>${job.stopping ? "stopping" : job.status}${
              job.status === "running" ? ` &middot; ${Math.round(job.fraction * 100)}%` : ""
            }</span>
          </div>
          <div class="bar"><i style="width:${Math.round(job.fraction * 100)}%"></i></div>
          <div class="job-msg">${job.error ?? job.message}</div>
          ${
            job.result?.failed?.length
              ? `<div class="job-msg" style="color:var(--bad)">${job.result.failed
                  .map((f) => `${f.case}: ${f.error}`)
                  .join("<br>")}</div>`
              : ""
          }
          ${s?.tiles ? `<div class="job-msg">${pct(s.tile_agreement)} agreement over ${s.tiles} tiles.</div>` : ""}
        </div>`;
      })
      .join("");

    if (!mine.some((job) => job.status === "running" || job.status === "queued")) {
      clearInterval(state.sixslides.polling);
      state.sixslides.polling = null;
      loadSixSlides();
    }
  };

  await tick();
  state.sixslides.polling = setInterval(tick, 1500);
}

$("#ss-run").onclick = () => runSixSlides(null);
$("#ss-run-missing").onclick = () =>
  runSixSlides(state.sixslides.items.filter((i) => i.available && !i.result).map((i) => i.case));
