/* The whole-slide screen: both models against a pathologist's own boxes.

   Loaded after app.js, so `$`, `api`, `pct`, `state` are already in the shared global
   lexical scope. Every number is computed on the server and printed verbatim; nothing
   here re-derives a score. */

const WSI_CLASSES = ["non_epithelium", "non_invasive_epithelium", "invasive_epithelium"];
const WSI_SHORT = { non_epithelium: "other", non_invasive_epithelium: "in-situ", invasive_epithelium: "invasive" };

state.wsi = { items: [], polling: null };

async function loadWsi() {
  let data;
  try {
    data = await api("/wsi");
  } catch (error) {
    $("#wsi-status").textContent = "Could not list the slides: " + error.message;
    return;
  }
  state.wsi.items = data.items;

  const caseSelect = $("#wsi-case");
  if (!caseSelect.options.length) {
    caseSelect.innerHTML = data.items.map((i) => `<option value="${i.case}">${i.case}</option>`).join("");
  }
  const modelSelect = $("#wsi-model");
  if (!modelSelect.options.length && data.models.length) {
    modelSelect.innerHTML = data.models
      .map((m) => `<option value="${m}"${m === data.default_model ? " selected" : ""}>${m}</option>`)
      .join("");
  }

  if (!data.items.length) {
    $("#wsi-status").innerHTML =
      "No slide with matching annotations found in <code>Testing_H&amp;E_images_BRACS/</code>.";
    $("#wsi-body").innerHTML = "";
    return;
  }
  renderWsi(data.items.find((i) => i.case === caseSelect.value) ?? data.items[0]);
}

function truthSummary(entry) {
  const preview = entry.result?.annotations ?? entry.annotation_preview;
  if (!preview) return "";
  const rows = Object.entries(preview.by_class)
    .map(([name, v]) => `<tr><td>${name}</td><td>${v.boxes} boxes</td><td>${v.mm2} mm²</td></tr>`)
    .join("");
  return `<table class="kv" style="max-width:360px">
    <tr><td><b>the pathologist drew</b></td><td>${preview.boxes} boxes</td><td>${preview.total_mm2} mm²</td></tr>
    ${rows}
  </table>`;
}

function scoreCard(who, block, title, subtitle) {
  if (!block) return "";
  const head = WSI_CLASSES.map((c) => `<th>${WSI_SHORT[c]}</th>`).join("");
  const rows = block.confusion
    .map((row, i) => {
      const cells = row
        .map((v, j) => `<td class="${i === j ? "diag" : v > 0 ? "bad" : ""}">${v}</td>`)
        .join("");
      return `<tr><th style="text-align:left">${WSI_SHORT[WSI_CLASSES[i]]}</th>${cells}</tr>`;
    })
    .join("");
  const r = block.recall;
  return `<div class="card-box">
    <h3>${title}</h3>
    <p class="note" style="margin:0 0 .5rem">${subtitle}</p>
    <div class="bigstat" style="margin-bottom:.6rem">
      <div><b>${pct(block.accuracy)}</b><span>of annotated tiles correct</span></div>
      <div><b style="color:var(--c1)">${r.non_invasive_epithelium == null ? "—" : pct(r.non_invasive_epithelium)}</b>
           <span>in-situ found</span></div>
      <div><b style="color:var(--c2)">${r.invasive_epithelium == null ? "—" : pct(r.invasive_epithelium)}</b>
           <span>invasive found</span></div>
    </div>
    <table class="cm">
      <tr><th style="text-align:left">pathologist &darr; / ${who} &rarr;</th>${head}</tr>
      ${rows}
    </table>
    <p class="note" style="margin:.5rem 0 0">
      in-situ called invasive <b>${block.in_situ_called_invasive}</b> ·
      invasive called in-situ <b>${block.invasive_called_in_situ}</b>
    </p>
  </div>`;
}

function renderWsi(entry) {
  const body = $("#wsi-body");
  if (!entry) { body.innerHTML = ""; return; }

  if (!entry.result) {
    $("#wsi-status").innerHTML =
      `<b>${entry.case}</b> — ${(entry.bytes / 1e9).toFixed(2)} GB, not segmented yet. ` +
      `The whole tissue takes about an hour on one BEETLE fold.` +
      (entry.annotation_error ? ` <span style="color:var(--bad)">${entry.annotation_error}</span>` : "");
    body.innerHTML = `<div class="card-box">
      <h3>What is waiting to be checked</h3>
      ${truthSummary(entry)}
      <p class="note" style="margin:.6rem 0 0">
        Press <b>Segment the whole tissue</b> to run both models over all of it.
      </p>
    </div>`;
    return;
  }

  const r = entry.result;
  const s = r.scored;
  const url = (which) => `/api/wsi/${entry.case}/image/${which}?v=${encodeURIComponent(r.seconds)}`;

  $("#wsi-status").innerHTML =
    `<b>${entry.case}</b> — ${r.slide_size[0]}×${r.slide_size[1]} px at ${r.slide_mpp} µm/px, ` +
    `${r.tissue_mm2} mm² of tissue, <b>${r.tiles_tissue.toLocaleString()}</b> tiles segmented ` +
    `by each model in ${(r.seconds / 60).toFixed(0)} min. ` +
    `<b>${s.annotated_tiles}</b> of them fall inside a pathologist's box and are scored.`;

  body.innerHTML = `
    <div class="matrix">
      <div class="matrix-head" style="grid-template-columns: repeat(4, minmax(0,1fr))">
        <span><b>the tissue</b></span>
        <span><b>the pathologist</b> — the right answer</span>
        <span><b>our model</b></span>
        <span><b>BEETLE</b></span>
      </div>
      <div class="matrix-row" style="grid-template-columns: repeat(4, minmax(0,1fr))">
        <div class="cell"><img src="${url("original")}" alt="the tissue"><span class="tag">the tissue</span></div>
        <div class="cell"><img src="${url("truth")}" alt="pathologist"><span class="tag">pathologist · ${r.annotations.boxes} boxes</span></div>
        <div class="cell blocky"><img src="${url("ours")}" alt="our model"><span class="tag">our model</span></div>
        <div class="cell blocky"><img src="${url("beetle")}" alt="BEETLE"><span class="tag">BEETLE</span></div>
      </div>
    </div>

    <div class="cols" style="grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem">
      ${scoreCard("ours", s.ours, "Our model vs the pathologist",
        "Trained on BCSS plus BEETLE-labelled BRACS regions. Has never seen this slide.")}
      ${scoreCard("BEETLE", s.beetle, "BEETLE vs the pathologist",
        "The model that produced our training labels. Also has never seen this slide.")}
    </div>

    <div class="card-box">
      <h3>What the pathologist annotated</h3>
      ${truthSummary(entry)}
      <details>
        <summary>How BRACS's labels were mapped, and what was left unscored</summary>
        <div class="detail-body">
          <table class="kv" style="max-width:420px">
            ${Object.entries(r.bracs_to_ours)
              .map(([k, v]) => `<tr><td><code>${k}</code></td><td>${WSI_SHORT[WSI_CLASSES[v]]}</td></tr>`)
              .join("")}
          </table>
          <p style="margin:.6rem 0 0">${r.unlabelled_rule}</p>
          <p style="margin:.4rem 0 0">
            ${(r.tiles_tissue - s.annotated_tiles).toLocaleString()} tissue tiles are outside
            every box and were segmented but not scored.
          </p>
        </div>
      </details>
    </div>`;
}

$("#wsi-case").onchange = () =>
  renderWsi(state.wsi.items.find((i) => i.case === $("#wsi-case").value));

$("#wsi-run").onclick = async () => {
  try {
    await api("/wsi/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case: $("#wsi-case").value,
        folds: $("#wsi-folds").value.split(",").map(Number),
        model_name: $("#wsi-model").value || null,
      }),
    });
  } catch (error) {
    alert("Could not start: " + error.message);
    return;
  }
  pollWsi();
};

async function pollWsi() {
  if (state.wsi.polling) clearInterval(state.wsi.polling);
  const tick = async () => {
    let data;
    try { data = await api("/jobs?limit=8"); } catch { return; }
    const mine = data.items.filter((job) => job.kind === "wsi");
    $("#wsi-jobs").innerHTML = mine
      .slice(0, 2)
      .map((job) => `<div class="job ${job.status}">
        <div class="job-head"><b>${job.label}</b>
          <span>${job.status}${job.status === "running" ? ` · ${Math.round(job.fraction * 100)}%` : ""}</span>
        </div>
        <div class="bar"><i style="width:${Math.round(job.fraction * 100)}%"></i></div>
        <div class="job-msg">${job.error ?? job.message}</div>
      </div>`)
      .join("");

    if (!mine.some((job) => job.status === "running" || job.status === "queued")) {
      clearInterval(state.wsi.polling);
      state.wsi.polling = null;
      loadWsi();
    }
  };
  await tick();
  state.wsi.polling = setInterval(tick, 3000);
}
