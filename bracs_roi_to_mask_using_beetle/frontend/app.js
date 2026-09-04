/* The whole frontend: one screen, no framework, no build step.

   It used to be four screens - a picker over the 665 BRACS regions, a mask viewer, a
   stain comparison, and this. Those three drove the experiment and are now driven from
   `scripts/batch.py` and `scripts/export_to_approach1.py` instead; the endpoints they
   used are still there and still tested. What is left here is the one thing worth
   having a page for: give the model an image, see what it makes of it.

   One rule followed throughout: every number shown here is computed on the server and
   rendered verbatim. Nothing is recalculated in the browser - a verdict that the UI
   works out for itself is a second definition of "did it work", and the two would
   disagree eventually. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  settings: null,
  sixslides: {
    items: [],
    // Its OWN interval handle. The upload screen owns `state.upload.polling` and clears
    // it when its own job drains; one shared handle would have each loop cancelling the
    // other the moment both screens had been visited.
    polling: null,
  },
  upload: {
    file: null,       // what is in the file input, not yet sent
    current: null,    // the upload id being shown
    previewUrl: null, // blob: URL for the local file, until the server's copy arrives
    items: [],
    polling: null,
  },
};

const pct = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");
const CLASS_LABEL = {
  non_epithelium: "Other tissue",
  non_invasive_epithelium: "In-situ (DCIS)",
  invasive_epithelium: "Invasive",
};
const CLASS_VAR = {
  non_epithelium: "--c0",
  non_invasive_epithelium: "--c1",
  invasive_epithelium: "--c2",
};

async function api(path, options) {
  const response = await fetch("/api" + path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return response.json();
}

/* --- health -------------------------------------------------------------- */

/* `/api/health` reports every input the project needs, `bracs_regions` among them,
   and `health.ok` is an AND over all of them. This page reads none of the BRACS
   regions, so a chip saying "missing: bracs_regions" over a perfectly working upload
   form would be false. The three checks this page actually stands on are named here;
   the endpoint keeps reporting the rest for the scripts that do need them. */
const NEEDED_HERE = ["beetle_weights", "approach1_src", "demo_backend"];

async function loadHealth() {
  const box = $("#health");
  try {
    const health = await api("/health");
    const broken = NEEDED_HERE.filter((check) => !health.checks[check]);
    const ready = broken.length === 0;
    box.className = "health " + (ready ? "ok" : "bad");
    box.textContent = ready
      ? `ready · teacher at ${health.teacher.teacher_mpp} µm/px`
      : "missing: " + broken.join(", ");
    box.title = ready
      ? "BEETLE's weights and approach 1's exporter are present."
      : JSON.stringify(health, null, 2);
  } catch (error) {
    box.className = "health bad";
    box.textContent = "backend unreachable";
  }
}

/* --- settings ------------------------------------------------------------ */

async function loadSettings() {
  state.settings = await api("/settings");
  // Pre-filled with BRACS's own assumed resolution, as a starting point rather than a
  // default: for an image somebody uploads, nothing on disk supports even that guess.
  // `config.BRACS_MPP` says at length what getting it wrong costs.
  $("#up-mpp").value = state.settings.bracs_mpp;
}

/* --- shared bits of the readout ------------------------------------------ */

function bars(fractions) {
  return (
    '<div class="bars">' +
    Object.entries(CLASS_LABEL)
      .map(([key, label]) => {
        const value = fractions[key] ?? 0;
        return `<div class="row">
          <span class="name">${label}</span>
          <span class="track"><i style="width:${(100 * value).toFixed(1)}%;background:var(${CLASS_VAR[key]})"></i></span>
          <span class="val">${pct(value)}</span>
        </div>`;
      })
      .join("") +
    "</div>"
  );
}

/* "Other tissue" is four of BEETLE's classes collapsed into one, and on a comedo DCIS
   duct most of it is dead cells rather than supporting tissue - which is the single most
   diagnostically interesting thing in such an image, and calling it "other" hides it.
   The split is already in the manifest, so showing it costs a lookup.

   Only shown when necrosis is actually present. A permanent row reading "necrosis 0.0%"
   would train the reader to ignore the line on the images where it matters. */
function whatIsOtherTissue(manifest) {
  const areas = manifest.teacher?.beetle_class_area ?? {};
  const necrosis = areas.necrosis ?? 0;
  if (necrosis < 0.02) return "";

  const stromaEtc = areas.other ?? 0;
  return `
    <p class="note" style="margin:.7rem 0 0">
      <b>${pct(necrosis)}</b> of this region is <b>dead tissue</b> (necrosis) and
      ${pct(stromaEtc)} is supporting tissue. Both are counted as "other tissue" above
      because neither is scored — but a duct packed with dead cells in the middle and
      living cancer cells round the rim is <em>comedo</em> DCIS, and that pattern is a
      marker of the high-grade form.
    </p>
    <details>
      <summary>Why they are merged</summary>
      <div class="detail-body">
        <p>
          The tissue model answers one question — is this tile invasive carcinoma — so
          stroma, fat, inflammation, blood and necrosis all collapse into class 0. That
          is a deliberate modelling decision, not a limitation of the segmentation:
          BEETLE reported these separately and
          <code>manifest.json → teacher.beetle_class_area</code> keeps them apart.
        </p>
        <p>
          It does mean these masks must never be reused by anything that needs necrosis
          told apart from stroma — the written mask has necrosis as
          <code>necrosis_or_debris</code> (4) and everything else as
          <code>stroma</code> (2), so that distinction survives in the mask even though
          it does not survive the tile vote.
        </p>
      </div>
    </details>`;
}

/* --- the screen ---------------------------------------------------------- */

/* What each colour on the overlay means, in the words somebody who is not a
   pathologist can check the picture against. The colours, the class grouping and the
   codes all come from the server (`labels.colour_legend`) so they cannot drift from the
   pixels; only these sentences live here, because they are prose and there is nowhere
   better for them.

   The second half of the DCIS sentence is the one that matters and is the one everybody
   omits: BEETLE's non-invasive class is its own documented mix of "healthy glands and
   DCIS ... LCIS, atypical ductal hyperplasia, apocrine metaplasia". Amber therefore does
   not mean cancer. On a BRACS DCIS region that distinction is carried by the dataset's
   annotation; on an arbitrary upload nothing carries it, so it has to be said. */
const COLOUR_MEANING = {
  non_invasive_epithelium: {
    title: "Gland lining that has not broken out",
    text:
      "Cancer cells still contained inside a duct — DCIS — <b>and also ordinary healthy " +
      "gland lining</b>, which this model does not tell apart from it. Amber means " +
      "“lining, still inside”, not “cancer”.",
  },
  invasive_epithelium: {
    title: "Cancer that has broken out",
    text:
      "Cells that have left the duct and are growing into the surrounding tissue. This " +
      "is the distinction the whole project exists to draw.",
  },
  non_epithelium: {
    title: "Everything else",
    text:
      "Supporting tissue, fat, inflammation, blood — and dead tissue. Four separate " +
      "things the model reports separately and this picture merges, because none of " +
      "them is scored.",
  },
  unlabelled: {
    title: "Not tissue",
    text:
      "Glass, background, or the blank edge of the crop. Left uncoloured on purpose: " +
      "tinting it would imply the model said something there.",
  },
};

const uploadEl = {
  file: () => $("#up-file"),
  run: () => $("#up-run"),
  status: () => $("#up-status"),
};

uploadEl.file().onchange = () => {
  const file = uploadEl.file().files?.[0] ?? null;
  state.upload.file = file;
  const run = uploadEl.run();
  run.disabled = !file;
  run.textContent = file ? "Segment this image" : "Choose an image first";
  uploadEl.status().className = "note";
  uploadEl.status().innerHTML = file
    ? `<b>${file.name}</b> — ${(file.size / 1e6).toFixed(1)} MB. ` +
      "Check the resolution on the right before running: it decides how big the model " +
      "thinks the cells are, and it cannot detect a wrong answer."
    : "Nothing chosen yet.";
};

uploadEl.run().onclick = async () => {
  const file = state.upload.file;
  if (!file) return;

  const body = new FormData();
  body.append("file", file);
  body.append("source_mpp", $("#up-mpp").value);
  body.append("folds", $("#up-folds").value);

  const run = uploadEl.run();
  run.disabled = true;
  run.textContent = "Uploading…";
  try {
    const started = await api("/uploads", { method: "POST", body });
    state.upload.current = started.upload_id;
    uploadEl.status().className = "note";
    uploadEl.status().textContent =
      `${started.upload.filename} — ${started.upload.native_size.join(" × ")} px, ` +
      `read as ${started.upload.source_mpp} µm/px.`;
    // Shown straight away from the local file, so there is something to look at during
    // the minute the forward pass takes. Replaced by the server's resampled copy when
    // the result lands - see `showUpload`.
    $("#up-pair").hidden = false;
    if (state.upload.previewUrl) URL.revokeObjectURL(state.upload.previewUrl);
    state.upload.previewUrl = URL.createObjectURL(file);
    $("#up-original").src = state.upload.previewUrl;
    $("#up-under").src = state.upload.previewUrl;
    $("#up-overlay").removeAttribute("src");
    $("#up-details").innerHTML = "";
    pollUploadJob(started.job.id, started.upload_id);
  } catch (error) {
    uploadEl.status().className = "note warn";
    uploadEl.status().textContent = "Could not upload: " + error.message;
  } finally {
    run.disabled = !state.upload.file;
    run.textContent = state.upload.file ? "Segment this image" : "Choose an image first";
  }
};

/* Polls this one job by id rather than the recent-jobs list. `jobs.RUNNER` is shared
   with `scripts/batch.py`, so a batch of 40 BRACS regions can be queued behind or in
   front of this upload, and somebody watching their own image should not have that
   queue's progress scrolling past underneath it. */
async function pollUploadJob(jobId, uploadId) {
  if (state.upload.polling) clearInterval(state.upload.polling);
  state.upload.polling = null;
  const box = $("#up-job");
  // A one-fold run on a small crop can finish inside the first `tick`, so whether to
  // start an interval at all is tracked here rather than read back off
  // `state.upload.polling` - which is null both before the interval exists and after
  // the job cleared it.
  let finished = false;

  const tick = async () => {
    let job;
    try { job = await api(`/jobs/${jobId}`); } catch { return; }
    const done = ["done", "failed", "cancelled"].includes(job.status);
    box.innerHTML = `
      <div class="job ${job.status}">
        <div class="job-head">
          <b>${job.label}</b>
          <span>${job.status}${job.status === "running" ? ` · ${Math.round(job.fraction * 100)}%` : ""}</span>
        </div>
        <div class="bar"><i style="width:${Math.round(job.fraction * 100)}%"></i></div>
        <div class="job-msg">${job.error ?? job.message}</div>
      </div>`;

    if (!done) return;
    finished = true;
    if (state.upload.polling) {
      clearInterval(state.upload.polling);
      state.upload.polling = null;
    }
    if (job.status === "done") await showUpload(uploadId);
    await loadUploads({ keep: true });
  };

  await tick();
  if (!finished) state.upload.polling = setInterval(tick, 1200);
}

function swatches(legend, areas) {
  return (
    '<div class="swatches">' +
    legend
      .map((entry) => {
        const meaning = COLOUR_MEANING[entry.class_name] ?? { title: entry.class_name, text: "" };
        const area = areas?.[entry.class_name];
        return `
        <div class="swatch ${entry.scored ? "" : "muted"}">
          <span class="blob" style="background:${entry.hex}"></span>
          <div>
            <div class="who">${meaning.title}</div>
            <div class="what">${meaning.text}</div>
          </div>
          <div class="area">${area == null ? "—" : pct(area)}<small>of the picture</small></div>
        </div>`;
      })
      .join("") +
    "</div>"
  );
}

/* Draw one finished upload: the server's own resampled copy in the left box, the
   overlay in the right, and what the colours mean underneath.

   The left box switches from the browser's preview of the local file to
   `image/original` on purpose. That endpoint serves the *resampled* image, which is
   what the model actually looked at - and if the resolution was set wrong, the two boxes
   being visibly the same size is the only on-screen sign that the scaling happened at
   all. A before/after pair where "before" is the untouched upload and "after" is a
   half-size mask would hide the one mistake this screen is most likely to contain. */
async function showUpload(uploadId) {
  let record;
  try { record = await api(`/uploads/${uploadId}`); } catch (error) {
    uploadEl.status().className = "note warn";
    uploadEl.status().textContent = "Could not read the result: " + error.message;
    return;
  }
  state.upload.current = uploadId;
  const manifest = record.manifest;

  /* Nothing is pointed at an endpoint until the files behind it exist. All three of
     them 404 while a run is going, and an <img> whose src 404s is exactly where the
     browser draws its broken-image glyph - so a click on a still-running upload in the
     history strip used to fill both panes with that. The pair stays hidden instead and
     the job is picked back up, so the pictures appear when there are pictures. */
  if (!record.ready) {
    $("#up-pair").hidden = true;
    $("#up-details").innerHTML = "";
    uploadEl.status().className = "note";
    uploadEl.status().textContent = `${record.filename} is still being segmented.`;
    if (record.job_id) pollUploadJob(record.job_id, uploadId);
    return;
  }

  // The local-file preview has done its job; the panes are about to point at the
  // server's resampled copy instead.
  if (state.upload.previewUrl) {
    URL.revokeObjectURL(state.upload.previewUrl);
    state.upload.previewUrl = null;
  }
  $("#up-pair").hidden = false;

  // Not cache-busted: an upload's pixels never change once written, so a fresh URL per
  // render would re-download a megabyte on every click through the history strip.
  const original = `/api/uploads/${uploadId}/image/original`;
  $("#up-original").src = original;
  $("#up-under").src = original;

  if (!manifest) {
    $("#up-src-note").textContent = "";
    $("#up-overlay").removeAttribute("src");
    $("#up-details").innerHTML =
      '<p class="note">This upload has no result on disk. It may have failed; try it again.</p>';
    return;
  }

  $("#up-overlay").src = `/api/uploads/${uploadId}/image/overlay`;
  $("#up-overlay").style.opacity = $("#up-fade").value / 100;
  $("#up-src-note").textContent =
    `${manifest.native_size.join(" × ")} px at ${manifest.source_mpp} µm/px`;

  const areas = manifest.class_area_fraction;
  const legend = state.settings?.colour_legend ?? [];

  $("#up-details").innerHTML = `
    <div class="cols">
      <div>
        <div class="card-box">
          <h3>What the colours mean</h3>
          ${swatches(legend, areas)}
          <details>
            <summary>Which of the model's own classes each colour holds</summary>
            <div class="detail-body">
              <p>
                BEETLE reports five classes. They are merged into three here because the
                tissue model this feeds answers one question — is this invasive
                carcinoma — and the merge is done by approach 1's own
                <code>bcss.remap</code>, not by this page. The number beside each is the
                BCSS label code written into the mask file, which keeps distinctions the
                colours do not.
              </p>
              <table class="kv">
                ${legend
                  .map(
                    (entry) => `<tr>
                      <td><span class="blob" style="display:inline-block;width:.7rem;height:.7rem;border-radius:3px;background:${entry.hex};vertical-align:-1px;margin-right:.4rem"></span>${entry.class_name}</td>
                      <td>${
                        Object.entries(entry.bcss_codes)
                          .map(([name, code]) => `<code>${name}</code> → ${code}`)
                          .join("<br>") || "—"
                      }</td>
                    </tr>`
                  )
                  .join("")}
              </table>
            </div>
          </details>
        </div>

        <div class="card-box">
          <h3>How much of each</h3>
          ${bars(areas)}
          ${whatIsOtherTissue(manifest)}
        </div>
      </div>

      <div>
        <div class="card-box">
          <h3>Is this right?</h3>
          <p class="note" style="margin-top:0">
            This app cannot tell you. Nobody has said what is in this image, so there
            is nothing for the numbers to be checked against — which is why there is no
            green or red verdict here. The batch pipeline does produce one, but only
            because BRACS has already annotated every region it runs on as DCIS, and it
            can therefore catch the model disagreeing with that. Here it cannot.
          </p>
          <p class="note">
            The check you <em>can</em> make is the one a pathologist would: drag the
            slider and see whether the amber follows the outlines of the ducts in the
            tissue underneath, rather than spilling across them.
          </p>
        </div>

        <div class="card-box">
          <h3>What was run</h3>
          <table class="kv">
            <tr><td>Models used</td><td>${manifest.teacher.folds.length} of 5</td></tr>
            <tr><td>Model's confidence</td><td>${pct(manifest.teacher.mean_confidence)}</td></tr>
            ${
              manifest.teacher.mean_fold_disagreement != null
                ? `<tr><td>Models disagreed by</td><td>${pct(manifest.teacher.mean_fold_disagreement)}</td></tr>`
                : `<tr><td>Models disagreed by</td><td title="A single model cannot disagree with itself.">n/a — one model</td></tr>`
            }
            <tr><td>You uploaded</td><td>${manifest.native_size.join(" × ")}</td></tr>
            <tr><td>The model saw</td><td>${manifest.working_size.join(" × ")}</td></tr>
            <tr><td>Resolution</td><td>${manifest.source_mpp} → ${manifest.working_mpp} µm/px</td></tr>
            <tr><td>Took</td><td>${manifest.seconds}s</td></tr>
          </table>
          <p class="note" style="margin-bottom:0">
            Confidence is how peaked the model's output was, not how often it is right.
            A model shown tissue outside what it was trained on is frequently confident
            and wrong.
          </p>
          <details>
            <summary>The mask file itself</summary>
            <div class="detail-body">
              <p>
                <a href="/api/uploads/${uploadId}/image/mask" download style="color:var(--accent)">
                  Download the mask</a> — single-channel uint16 PNG carrying the BCSS
                label codes in the table on the left, at the working size above. It is
                not a picture: opening it in a viewer shows near-black, because the
                values are 0, 1, 2, 4 and 20 out of 65535.
              </p>
              <p>
                It pairs with
                <a href="/api/uploads/${uploadId}/image/input" download style="color:var(--accent)">the
                image the model actually read</a> — your upload resampled to
                ${manifest.working_mpp} µm/px — and only with that one. The picture in
                the box above is capped at 1400 px for the browser, so on a larger
                upload it does not line up with the mask pixel for pixel.
              </p>
              <p>
                No training tiles were cut from it. An uploaded image carries no
                annotation saying what it contains, so tiles from it would be training
                data labelled by a guess about an unknown specimen.
              </p>
            </div>
          </details>
        </div>
      </div>
    </div>`;
}

$("#up-fade").oninput = (event) => {
  const overlay = $("#up-overlay");
  if (overlay) overlay.style.opacity = event.target.value / 100;
};

async function loadUploads(options = {}) {
  let data;
  try { data = await api("/uploads"); } catch { return; }
  state.upload.items = data.items;

  const box = $("#up-history-box");
  box.hidden = !data.items.length;
  $("#up-history").innerHTML = data.items
    .map(
      (item) => `
      <div class="up ${item.upload_id === state.upload.current ? "sel" : ""}" data-id="${item.upload_id}">
        ${
          item.ready
            ? `<img loading="lazy" src="/api/uploads/${item.upload_id}/image/overlay" alt="">`
            : '<div class="pending">still working…</div>'
        }
        <button class="kill" data-kill="${item.upload_id}" title="Delete this upload">×</button>
        <span title="${item.filename}">${item.filename}</span>
      </div>`
    )
    .join("");

  $$("#up-history .up").forEach((card) => {
    card.onclick = (event) => {
      if (event.target.dataset.kill) return;
      showUpload(card.dataset.id);
      $$("#up-history .up").forEach((other) => other.classList.toggle("sel", other === card));
    };
  });
  $$("#up-history .kill").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      const id = button.dataset.kill;
      try { await api(`/uploads/${id}`, { method: "DELETE" }); } catch (error) {
        alert("Could not delete: " + error.message);
        return;
      }
      if (state.upload.current === id) {
        state.upload.current = null;
        $("#up-pair").hidden = true;
        $("#up-details").innerHTML = "";
      }
      loadUploads();
    };
  });

  // Nothing on screen and something on disk: show the newest, so a reload does not
  // present two empty boxes next to a strip full of finished results. A newest upload
  // that is still running is handed over too - `showUpload` picks its job back up
  // rather than rendering it.
  if (!options.keep && !state.upload.current && data.items.length) {
    showUpload(data.items[0].upload_id);
  }
}

/* --- tabs ---------------------------------------------------------------- */

/* Two panels and no router. Adding a screen is one button, one section, and one line
   here - the same mechanism the app had before it was cut down to a single screen. */
$$(".tab").forEach((tab) => {
  tab.onclick = () => {
    $$(".tab").forEach((other) => other.classList.toggle("active", other === tab));
    $$(".panel").forEach((panel) =>
      panel.classList.toggle("active", panel.id === tab.dataset.panel));
    if (tab.dataset.panel === "panel-sixslides") loadSixSlides();
    if (tab.dataset.panel === "panel-wsi") loadWsi();
  };
});

/* --- boot ---------------------------------------------------------------- */

(async function start() {
  await loadHealth();
  // Before `loadUploads`, not alongside it: `showUpload` reads `colour_legend` off
  // `state.settings` to label the swatches, and an upload rendered without it would
  // come up with an unlabelled legend.
  await loadSettings();
  await loadUploads();
  // The six-slide screen is the one that opens, so it loads last but is not awaited
  // behind the upload history: a slow `sixslides` listing must not leave the visible
  // panel blank while an invisible one populates.
  loadSixSlides();
  // The pathologist screen is the one that opens.
  loadWsi();
})();
