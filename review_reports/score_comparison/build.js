// AI scores (pixel-wise and tile-based) against the four pathologists: analysis and per-slide review.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, HeadingLevel, BorderStyle, ShadingType,
  Footer, PageNumber, LevelFormat, VerticalAlign,
} = require("docx");

// Reads what the Python steps wrote under v<N>_data/data/review_reports/score_comparison.
// The version is asked of data_versions.py, so it follows the same rule as the app:
// CSS_DATA_VERSION, then storage/ACTIVE_DATA_VERSION, falling back to the latest
// version this code can open.
const ROOT = path.join(__dirname, "..", "..");
function dataVersion() {
  const script = path.join(ROOT, "data_versions.py");
  for (const python of [process.env.PYTHON, "python", "py"].filter(Boolean)) {
    try {
      const out = require("child_process")
        .execFileSync(python, [script, "--active"], { encoding: "utf8" })
        .trim();
      if (/^v[1-9][0-9]*$/.test(out)) return out;
    } catch (_) { /* try the next interpreter */ }
  }
  throw new Error("could not run data_versions.py to find the data version; set PYTHON");
}
const HERE = path.join(ROOT, "storage", `${dataVersion()}_data`, "data", "review_reports", "score_comparison");
const J = (f) => JSON.parse(fs.readFileSync(path.join(HERE, f), "utf8"));
const rows = J("rows.json"), stats = J("stats.json"), an = J("analysis.json"), slides = J("slides.json");
const OUT = process.argv[2];

const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 800;
const CONTENT = PAGE_W - 2 * MARGIN; // 10306
const FONT = "Calibri";
const HEAD = "DCE6F2", SUB = "F2F2F2", WARN = "FFF4E5", RULE = "BFBFBF";
const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const borders = { top: border, bottom: border, left: border, right: border };
const MARKERS = ["A", "F", "R", "U", "W"], READERS = ["PS", "SA", "DP", "NK"];
const CASES = [...new Set(rows.map((r) => r.case))];
const NAME = Object.fromEntries(rows.map((r) => [r.marker, r.markerName]));
const COMPARTMENT = { A: "membrane", F: "membrane", R: "membrane", U: "cytoplasm", W: "cytoplasm" };

const n0 = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
const n2 = (v) => v.toFixed(2).replace(/0$/, "").replace(/\.0$/, "");
const sgn = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
const slide = (c, m, mode) => slides.find((s) => s.case === c && s.marker === m && s.mode === mode);
const reader = (c, m) => rows.find((r) => r.case === c && r.marker === m);

// --- primitives --------------------------------------------------------------------------------
function text(t, o = {}) { return new TextRun({ text: t, font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics, color: o.color }); }
function para(children, o = {}) {
  return new Paragraph({ children: Array.isArray(children) ? children : [text(children, o)], alignment: o.align,
    spacing: { before: o.before ?? 0, after: o.after ?? 100 }, keepNext: o.keepNext, keepLines: true, numbering: o.numbering });
}
const bullet = (children, o = {}) => para(children, { numbering: { reference: "bullets", level: 0 }, after: o.after ?? 70, keepNext: o.keepNext });
let listNo = 0;
const newList = () => ++listNo;
const numbered = (children, inst) => para(children, { numbering: { reference: "steps", level: 0, instance: inst }, after: 70 });
const h1 = (t, o = {}) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: o.pageBreak, keepNext: true,
  children: [text(t, { size: 30, bold: true })], spacing: { before: o.before ?? 240, after: 110 } });
const h2 = (t, o = {}) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, pageBreakBefore: o.pageBreak,
  children: [text(t, { size: 24, bold: true })], spacing: { before: o.before ?? 200, after: 90 } });
const B = (t) => text(t, { bold: true });

function cell(t, width, o = {}) {
  const runs = Array.isArray(t) ? t : [text(String(t), { size: o.size || 17, bold: o.bold, color: o.color })];
  const children = Array.isArray(o.paras) ? o.paras : [para(runs, { after: 0, align: o.align ?? AlignmentType.CENTER, keepNext: o.keepNext })];
  return new TableCell({ children, width: { size: width, type: WidthType.DXA }, borders, columnSpan: o.span,
    verticalAlign: o.valign ?? VerticalAlign.CENTER, shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: o.pad ?? 45, bottom: o.pad ?? 45, left: 70, right: 70 } });
}
function table(widths, headRows, bodyRows, o = {}) {
  const keep = o.keep;
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [...headRows.map((cells) => new TableRow({ tableHeader: true, cantSplit: true, children: cells })),
           ...bodyRows.map((cells) => new TableRow({ cantSplit: true, children: cells }))] });
}
function jpegSize(buf) {
  let i = 2;
  while (i < buf.length) {
    const marker = buf[i + 1], len = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7) };
    i += 2 + len;
  }
  throw new Error("not a JPEG");
}
function img(file, widthPx, boxH) {
  const buf = fs.readFileSync(file);
  const png = file.endsWith(".png");
  const { w, h } = png ? { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) } : jpegSize(buf);
  const limitH = boxH || Infinity;
  const scale = Math.min(widthPx / w, limitH / h);
  return new ImageRun({ type: png ? "png" : "jpg", data: buf, transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
    altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } });
}
const imgPara = (file, widthPx, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: o.keepNext ?? true, spacing: { before: o.before ?? 60, after: o.after ?? 40 }, children: [img(file, widthPx, o.box)] });
const caption = (t) => para([text(t, { size: 17, italics: true, color: "404040" })], { after: 180 });
const figure = (name, caption_) => [imgPara(path.join(HERE, name), 680), caption(caption_)];

// --- plain-words warning translation -----------------------------------------------------------
const GLOBAL = ["Provisional cut points", "Alignment machine-confirmed"];
function warnings(s) {
  return s.caveats.filter((c) => !GLOBAL.includes(c.title)).map((c) => {
    const t = c.title.toLowerCase();
    if (t.startsWith("alignment failed")) return { label: "Registration failed its own check", detail: c.text.replace(/\s*\(\d\)\s*/g, " • ").replace(/^ • /, "") };
    if (t.startsWith("denominator incomplete")) return { label: "Nuclei missed on the IHC slide", detail: c.text };
    if (t.startsWith("not a measurement")) return { label: "Too few cells to measure", detail: c.text };
    if (t.startsWith("thin denominator")) return { label: "Few cells counted", detail: c.text };
    if (t.startsWith("outside the expected range")) return { label: "Outside OncoStem's usual range", detail: c.text };
    if (t.startsWith("intensity near the edge")) return { label: "Intensity at the top of the band table", detail: c.text };
    if (t.startsWith("region weighting")) return { label: "Regions disagree with each other", detail: c.text };
    return { label: c.title, detail: c.text };
  });
}
const failedCheck = (s) => s.caveats.some((c) => c.title.toLowerCase().startsWith("alignment failed"));

// ================================================================================================
// PART 1 - SUMMARY
// ================================================================================================
const o = stats.overall, sb = an.scoreboard, ag = an.agreement;
const title = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [text("AI Scores vs Pathologists", { size: 50, bold: true })], spacing: { after: 100 } }),
  para([text("Analysis and slide-by-slide review · 6 cases · 5 markers · 4 pathologists · AI pixel-wise and tile-based", { size: 23, color: "404040" })], { after: 60 }),
  para([text(`Generated ${new Date().toISOString().slice(0, 10)} from results/oncostem_ai_scores.csv, results/oncostem_ai_scores_tiles.csv, 6Slide Reports2.xlsx and the stored pipeline runs in data/history.`, { size: 17, italics: true, color: "595959" })], { after: 160 }),
];

const L1 = newList(), L2 = newList(), L3 = newList();
const glance = [
  h1("The short answer", { before: 60 }),
  para([B("The AI scores are not yet reliable enough to replace or check a pathologist. "), text("They are computed correctly from what the pipeline measured, but four things upstream of the arithmetic are not right yet, and the first two fail the pipeline's own checks on nearly every slide.")]),
  bullet([B("Percent positive is typically " + o.pixel.pct.mae.toFixed(0) + " (pixel-wise) and " + o.tile.pct.mae.toFixed(0) + " (tile-based) points from the pathologists' mean. "), text(`The four pathologists are ${stats.readerMad.toFixed(1)} points from their own mean. The AI falls inside the pathologists' range on ${o.pixel.pct.within} and ${o.tile.pct.within} of 30 slides.`)]),
  bullet([B("Intensity reads about " + Math.abs(o.pixel.int.bias).toFixed(1) + " lower on the 0–2 scale, "), text(`inside the pathologists' range on ${o.pixel.int.within} and ${o.tile.int.within} of 30 slides.`)]),
  bullet([B("Pixel-wise and tile-based are statistically indistinguishable "), text(`(paired Wilcoxon p = ${ag.pixelVsTileWilcoxonP.toFixed(2)}). They differ from each other by ${stats.pixelVsTileMae.toFixed(0)} points on a typical slide.`)]),
  para([B("Why:")], { before: 80, after: 40 }),
  numbered([B("Nuclei are missed on the IHC slide. "), text(`On average the IHC slide yields ${Math.round(avgShort("pixel") * 100)}% (pixel-wise) and ${Math.round(avgShort("tile") * 100)}% (tile-based) fewer nuclei than the same regions on the H&E. Heavy brown stain hides the blue counterstain the nucleus detector reads; in the fields inspected, the cells lost are the stained tumour cells — see "Nuclei segmentation".`)], L1),
  numbered([B("Tumour cells are not reliably identified. "), text("The step that separates tumour cells from other cells works on nucleus size, and it marks itself untrustworthy on every stored run: the nuclei it receives average about 14 µm², fragments rather than whole tumour nuclei (about 45 µm² on the H&E).")], L1),
  numbered([B("The cut points were never fitted. "), text("The thresholds that decide which cells count as positive are provisional defaults. Refitting them on this sheet (tested on held-out cases) helps only modestly for percent, but clearly for intensity — see \"Can calibration or ML fix it?\".")], L1),
  numbered([B("Two registrations failed their own check. "), text("Both are CD44 (CAN_00267 and CAN_00865), where the IHC section holds far less tissue than the H&E. Every registration was confirmed by the batch run, not by a person.")], L1),
  para([B("Can ML fix it? "), text(`A regression model trained on the AI's own measurements reaches ${sb["ML recalibration, both methods (held-out)"].pctMae.toFixed(1)} points on held-out cases — but simply predicting each marker's usual pathologist score, with no AI at all, reaches ${sb["No-AI baseline (marker's usual reader score)"].pctMae.toFixed(1)}. So at present the AI adds real case-to-case information only for CD44, the one marker whose pathologist scores vary much between cases.`)], { before: 80 }),
];
function avgShort(mode) { const v = slides.filter((s) => s.mode === mode && s.densityShortfall != null).map((s) => Math.max(s.densityShortfall, 0)); return v.reduce((a, b) => a + b, 0) / v.length; }

// --- how the scores are calculated ---------------------------------------------------------------
const how = [
  h1("How the AI scores are calculated", { pageBreak: true, before: 0 }),
  para("Both files are produced by the same 19-step pipeline. The only difference between them is step 2: which outline of the invasive tumour the cells are counted inside."),
  numbered([B("Read and clean the H&E. "), text("Quality control removes pen, folds and blur; tissue is separated from glass; the stain colours are separated so each stain can be measured on its own.")], L2),
  numbered([B("Find the invasive tumour on the H&E. "), text("A tissue classifier labels every square tile of the slide. "), B("Tile-based file: "), text("the invasive tumour is the union of the tiles it called invasive (blocky edges). "), B("Pixel-wise file: "), text("inside the regions a person selected, BEETLE traces the invasive boundary pixel by pixel (tight edges, usually less area).")], L2),
  numbered([B("Carry the outline onto the IHC slide. "), text("The IHC section is aligned to the H&E with a transform that moves, turns and uniformly rescales it, chosen to maximise how much the two images agree (mutual information). The outline is moved with it.")], L2),
  numbered([B("Find the nuclei on the IHC slide. "), text("A deep-learning nucleus detector (InstanSeg) runs on the blue counterstain channel, on a sample of fields spread across each region — it does not segment every cell on the slide.")], L2),
  numbered([B("Keep the tumour cells. "), text("A rule on nucleus size, shape and darkness separates tumour cells from stromal and immune cells.")], L2),
  numbered([B("Measure the brown stain in the right part of each cell. "), text("CD44, ABCC4 and ABCC11 are membrane markers: a thin shell at the cell's edge is measured. N-cadherin and Pan-cadherin are cytoplasmic: the band around the nucleus is measured. The brown (DAB) is measured as optical density.")], L2),
  numbered([B("Call each cell positive or negative. "), text("Membrane markers: positive when the shell's mean brown density is at least 0.15 and at least 35% of the ring is stained. Cytoplasmic markers: at least 0.12 density and at least 25% of the band stained. These are the provisional cut points.")], L2),
  numbered([B("Percent positive "), text("= positive tumour cells ÷ all tumour cells, worked out per region, averaged over the regions by their area, then rounded to the nearest 5 — the way the pathologists report it.")], L2),
  numbered([B("Intensity "), text("= the mean brown density of the positive cells, mapped onto the 0–2 scale the pathologists use (0, 0.5, 1, 1.5, 1.75, 2). If no cell is positive the intensity is 0.")], L2),
  h2("What the other columns in the CSV files mean"),
  table([2600, 7706], [[cell("Column", 2600, { bold: true, fill: HEAD }), cell("Meaning", 7706, { bold: true, fill: HEAD })]], [
    ["percent, intensity, intensity_label", "The two delivered numbers, rounded to the pathologists' reporting steps."],
    ["percent_raw, intensity_raw", "The same before rounding; intensity_raw is the mean brown optical density of the positive cells."],
    ["percent_pooled, percent_plain_mean", "The percent worked out two other ways (all cells pooled; regions averaged without area weights), for comparison."],
    ["cells, positive_cells", "How many tumour cells were measured, and how many were called positive."],
    ["h_score, allred_*", "The same cells expressed on the H-score and Allred scales, for reference."],
    ["compartment, second_measure", "Membrane or cytoplasm, and whether ring completeness or stained fraction was used as the second condition."],
    ["cuts_provisional, caveats", "Whether the cut points are provisional (always, here), and the warnings the pipeline attached to that slide."],
  ].map(([a, b]) => [cell(a, 2600, { align: AlignmentType.LEFT, bold: true }), cell(b, 7706, { align: AlignmentType.LEFT })])),
  para([B("Is the arithmetic right? "), text(`Yes. Every percent in both files was recomputed from the stored per-cell measurements for this analysis and matches to within ${an.reproductionMaxAbs} of a point. The problems are in what was measured, not in how it was added up. ${an_stale()}`)], { before: 160 }),
];

function an_stale() {
  const st_ = J("attention.json").stale;
  return st_.length ? ` One bookkeeping error was found on the way: ${st_.length} row${st_.length === 1 ? "" : "s"} read stale files from an earlier run — see "Things that need attention", 1F.` : "";
}
// --- why zero ---------------------------------------------------------------------------------
const zeros = slides.filter((s) => s.percent <= 5 || s.intensity === 0).sort((a, b) => a.case.localeCompare(b.case) || a.marker.localeCompare(b.marker));
function zeroCause(s) {
  const bits = [];
  if (s.cells < 100) bits.push(`only ${s.cells} tumour cell${s.cells === 1 ? "" : "s"} were measured`);
  if (s.densityShortfall != null && s.densityShortfall > 0.5) bits.push(`${Math.round(s.densityShortfall * 100)}% of nuclei missed on the IHC slide`);
  if (failedCheck(s)) bits.push(`registration failed its check (IHC holds ${Math.round((s.tissueRatio || 0) * 100)}% of the H&E's tissue)`);
  if (s.carriedMm2 != null && s.carriedMm2 < 0.1) bits.push(`the outline covers only ${s.carriedMm2.toFixed(2)} mm²`);
  if (s.positive === 0 && s.cells > 0) bits.push("no measured cell cleared the provisional cut");
  return bits.join("; ") || "few positive cells at the provisional cut";
}
const Z = [1250, 1500, 900, 900, 800, 1000, 3956];
const whyZero = [
  h1("Why some scores are 0"),
  para([text("A 0 almost never means the tumour is negative. In every case below the pathologists scored the slide well above 0. A 0 appears when "), B("too few cells reach the measurement"), text(" (a tiny outline, nuclei the detector could not see, or an outline carried onto the wrong tissue) or when "), B("none of the measured cells clears the provisional cut"), text(". Intensity is 0 whenever percent is 0, because it is the average darkness of the positive cells and there are none.")]),
  table(Z, [[cell("Case", Z[0], { bold: true, fill: HEAD }), cell("Marker", Z[1], { bold: true, fill: HEAD }), cell("Method", Z[2], { bold: true, fill: HEAD }),
             cell("AI %", Z[3], { bold: true, fill: HEAD }), cell("Cells", Z[4], { bold: true, fill: HEAD }), cell("Pathologists", Z[5], { bold: true, fill: HEAD }), cell("Why", Z[6], { bold: true, fill: HEAD })]],
    zeros.map((s) => { const r = reader(s.case, s.marker); return [cell(s.case, Z[0]), cell(s.markerName, Z[1]), cell(s.mode === "pixel" ? "Pixel-wise" : "Tile-based", Z[2]),
      cell(`${n0(s.percent)}%`, Z[3], { bold: true }), cell(String(s.cells), Z[4]), cell(`${n0(r.readerPctMean)}%`, Z[5]), cell(zeroCause(s), Z[6], { align: AlignmentType.LEFT, size: 16 })]; })),
];

// --- reliability ------------------------------------------------------------------------------
const fl = an.flags;
const reliability = [
  h1("Is it reliable?", { pageBreak: true, before: 0 }),
  para("Not yet. Each reason below is measured, not assumed, and each is shown in pictures on the evidence pages."),
  h2("1. Nuclei are missed on the IHC slide — every slide, most slides badly"),
  para(`The same tumour sits on the H&E and on each IHC section, so the IHC should yield about as many nuclei per mm² as the H&E. It yields ${Math.round(avgShort("pixel") * 100)}% fewer on average (pixel-wise outlines; ${Math.round(avgShort("tile") * 100)}% tile-based). The warning is on ${fl.pixel.tally["Nuclei missed on the IHC slide"]} of 30 pixel-wise slides and ${fl.tile.tally["Nuclei missed on the IHC slide"]} of 30 tile-based ones. The detector reads the blue counterstain; where the brown stain is heavy the blue is masked. In the fields inspected the nuclei lost are the stained tumour cells, although across slides the shortfall does not track how much of the tumour is positive, so the direction of the bias on the final percent is not established. Figure 4 shows it slide by slide.`),
  h2("1b. Tumour cells are not reliably identified"),
  para("Step 14 keeps the tumour cells by nucleus size and shape, and checks that the nuclei it receives are big enough to be whole cells (median at least 25 µm²). It fails that check on all 101 stored runs: the median is about 14 µm², against about 45 µm² on the H&E. So the rules are sorting fragments, and the \"tumour cells\" behind every score are an estimate built on them."),
  h2("2. The cut points are provisional"),
  para("Every score in both files is flagged this way. The thresholds were set from the physics of the stain, not fitted to OncoStem's pathologists, because the reader sheet was not available when the scores were run. Refitting them is tested below: it helps, but it cannot recover cells the detector never found."),
  h2("3. Registration: machine-confirmed, and two failures"),
  para(`All ${slides.length} registrations were confirmed automatically by the batch run; no person has looked at them. Two failed the pipeline's own check and were let through so the marker would not drop out of the results: CD44 on CAN_00267 (image agreement 1.006, below the 1.008 floor, and the IHC holds 36% of the H&E's tissue) and CD44 on CAN_00865 (the IHC holds 19% of the H&E's tissue). Their scores are measured on whatever tissue the outline happened to land on. The companion document HE_IHC_Registration_Review.docx shows all 60 registrations.`),
  h2("4. Some slides are scored on very few cells"),
  para(`${fl.pixel.tally["Too few cells to measure"]} pixel-wise slides and ${fl.tile.tally["Too few cells to measure"]} tile-based slide were scored on so few cells that each cell moves the percent by several points or more; ${fl.pixel.tally["Few cells counted"] ?? fl.pixel.tally["Thin cell count"]} and ${fl.tile.tally["Thin cell count"]} more carry a thin-count warning. All of CAN_00267's pixel-wise slides are in this group: its BEETLE outline covers about 0.01 mm².`),
  h2("5. The AI's ranking of cases is only moderately right"),
  para(`Across all 30 slides the AI percent correlates with the pathologists' mean at Spearman ρ = ${ag.pixel.spearman} (pixel-wise) and ${ag.tile.spearman} (tile-based); Lin's concordance, which also penalises being off by a constant, is ${ag.pixel.ccc} and ${ag.tile.ccc}. The 95% limits of agreement (Bland–Altman) run from ${ag.pixel.baLow} to +${ag.pixel.baHigh} points pixel-wise and ${ag.tile.baLow} to +${ag.tile.baHigh} tile-based — a single slide can be off by 40–60 points in either direction. Per marker, the ranking is best for ABCC11 and CD44 and worst for ABCC4.`),
  h2("6. Two markers cannot be judged yet"),
  para("The pathologists recorded N-cadherin at 80% on every slide and Pan-cadherin at 75–80%. That looks like a reporting ceiling (\"80% or more\"). If it is, an AI score of 95–100% on those markers is not a disagreement, and part of the gap in Figures 1 and 3 is not real. This needs confirming with OncoStem."),
];

// ================================================================================================
// PART 2 - GRAPHS AND CALIBRATION
// ================================================================================================
const graphs = [
  h1("Scores side by side", { pageBreak: true, before: 0 }),
  para("Each panel is one marker, one column per case. The grey bar spans the four pathologists and the black tick is their mean. The blue circle is AI pixel-wise and the orange square is AI tile-based."),
  ...figure("fig_percent.png", "Figure 1. Percent positive cells. The AI sits above every pathologist range for ABCC4 and N-cadherin, and mostly below for CD44."),
  ...figure("fig_intensity.png", "Figure 2. Staining intensity (0–2). The AI mostly reads 1.0 where the pathologists read 1.25–2."),
  h1("Can calibration or ML fix it?", { pageBreak: true, before: 0 }),
  para("Three things were tried, each tested honestly: fitted on five cases and scored on the sixth, repeated so every case is held out once. A result on a case the model was fitted on is not reported as accuracy."),
  bullet([B("Refit the cut points. "), text("The pipeline's own calibration: for each marker, the positivity threshold that best matches the pathologists on the other five cases, then the intensity bands placed on the densities that threshold produces. Recomputed from the stored per-cell measurements; nothing is re-scanned.")]),
  bullet([B("ML recalibration. "), text("A ridge regression from both methods' slide-level measurements (raw percent, raw intensity, H-score, cell count) and the marker, predicting the pathologists' mean.")]),
  bullet([B("No-AI baseline. "), text("Predict each marker's usual pathologist score on the other five cases, ignoring the AI completely. Anything that cannot beat this is not using the slide.")]),
  ...figure("fig_scoreboard.png", "Figure 3. How far each approach lands from the pathologists (left) and how often its intensity falls inside their range (right)."),
];
const SBW = [3500, 1100, 900, 1150, 1100, 900, 1156];
const sbOrder = [["pixel: as delivered", "AI pixel-wise, as delivered"], ["tile: as delivered", "AI tile-based, as delivered"],
  ["pixel: cut refit (held-out)", "Pixel-wise, cut points refitted"], ["tile: cut refit (held-out)", "Tile-based, cut points refitted"],
  ["ML recalibration, both methods (held-out)", "ML recalibration of both"], ["No-AI baseline (marker's usual reader score)", "No AI: marker's usual score"]];
const calib = [
  table(SBW, [[cell("Approach", SBW[0], { bold: true, fill: HEAD }), cell("Percent: avg diff", SBW[1], { bold: true, fill: HEAD }), cell("Bias", SBW[2], { bold: true, fill: HEAD }),
               cell("In range", SBW[3], { bold: true, fill: HEAD }), cell("Intensity: avg diff", SBW[4], { bold: true, fill: HEAD }), cell("Bias", SBW[5], { bold: true, fill: HEAD }), cell("In range", SBW[6], { bold: true, fill: HEAD })]],
    sbOrder.map(([k, lab]) => { const v = sb[k]; return [cell(lab, SBW[0], { align: AlignmentType.LEFT }), cell(v.pctMae.toFixed(1), SBW[1], { bold: true }), cell(sgn(v.pctBias), SBW[2]),
      cell(`${v.inPct}/30`, SBW[3]), cell(v.intMae.toFixed(2), SBW[4], { bold: true }), cell(sgn(v.intBias, 2), SBW[5]), cell(`${v.inInt}/30`, SBW[6])]; })),
  para([B("Reading it. "), text(`Refitting the cut points brings percent from ${sb["pixel: as delivered"].pctMae.toFixed(1)} to ${sb["pixel: cut refit (held-out)"].pctMae.toFixed(1)} points pixel-wise and from ${sb["tile: as delivered"].pctMae.toFixed(1)} to ${sb["tile: cut refit (held-out)"].pctMae.toFixed(1)} tile-based, and brings intensity inside the pathologists' range on ${sb["pixel: cut refit (held-out)"].inInt} and ${sb["tile: cut refit (held-out)"].inInt} of 30 slides (from ${sb["pixel: as delivered"].inInt} and ${sb["tile: as delivered"].inInt}). That is the most useful fix available today, especially for intensity. The ML model's lower percent error is almost entirely the marker's usual score: the no-AI baseline is within a point of it.`)], { before: 140 }),
  h2("Per marker: does the AI carry information?"),
];
const PMW = [2300, 1300, 1300, 1300, 1300, 1400, 1406];
calib.push(table(PMW, [[cell("Marker", PMW[0], { bold: true, fill: HEAD }), cell("Pixel-wise as delivered", PMW[1], { bold: true, fill: HEAD }), cell("Tile-based as delivered", PMW[2], { bold: true, fill: HEAD }),
  cell("Pixel-wise refitted", PMW[3], { bold: true, fill: HEAD }), cell("Tile-based refitted", PMW[4], { bold: true, fill: HEAD }), cell("ML", PMW[5], { bold: true, fill: HEAD }), cell("No AI", PMW[6], { bold: true, fill: HEAD })]],
  MARKERS.map((m) => [cell(NAME[m], PMW[0], { align: AlignmentType.LEFT }), ...sbOrder.map(([k], i) => cell(sb[k].byMarker[m].pctMae.toFixed(1), PMW[i + 1], { bold: k.startsWith("ML") }))])));
calib.push(para("Average difference in percent points, held out where fitted. Only for CD44 does the ML model clearly beat the no-AI baseline: that is the marker where the slides genuinely differ, and the AI's measurements track the difference. For N-cadherin and Pan-cadherin the pathologists' scores are nearly constant, so the baseline is almost perfect and there is nothing for the AI to add.", { before: 100 }));
const RFW = [2300, 1400, 2150, 2150, 2306];
calib.push(h2("The refitted cut points", { pageBreak: true, before: 0 }));
calib.push(table(RFW, [[cell("Marker", RFW[0], { bold: true, fill: HEAD }), cell("Provisional cut", RFW[1], { bold: true, fill: HEAD }), cell("Fitted, pixel-wise", RFW[2], { bold: true, fill: HEAD }), cell("Fitted, tile-based", RFW[3], { bold: true, fill: HEAD }), cell("Stable across held-out folds?", RFW[4], { bold: true, fill: HEAD })]],
  MARKERS.map((m) => { const p = an.refit[`pixel:${m}`], t = an.refit[`tile:${m}`];
    const spread = (x) => { const v = Object.values(x.heldOut).filter(Boolean).map((h) => h.cut); return Math.max(...v) - Math.min(...v); };
    const stable = Math.max(spread(p), spread(t)) < 0.05 ? "yes" : `varies by up to ${Math.max(spread(p), spread(t)).toFixed(2)}`;
    return [cell(NAME[m], RFW[0], { align: AlignmentType.LEFT }), cell(p.provisionalCut.toFixed(2), RFW[1]), cell(p.cutAllCases.toFixed(3), RFW[2], { bold: true }), cell(t.cutAllCases.toFixed(3), RFW[3], { bold: true }), cell(stable, RFW[4])]; })));
calib.push(para("Brown optical density a cell must reach to count as positive. ABCC4's fitted cut is about three times the provisional one, which is why its AI score is 90–100% today. Where the cut moves a lot between folds, six cases are not enough to pin it down.", { before: 100 }));

// ================================================================================================
// PART 3 - VISUAL EVIDENCE
// ================================================================================================
function pairImages(s, px) {
  const W2 = CONTENT / 2;
  return table([W2, W2], [], [
    [cell([B(`H&E — invasive outline (${s.mode === "pixel" ? "pixel-wise" : "tile-based"})`)], W2, { fill: SUB, size: 16, keepNext: true }), cell([B("IHC — outline after registration")], W2, { fill: SUB, size: 16, keepNext: true })],
    [cell([], W2, { paras: [imgPara(s.heImage, px, { before: 0, after: 0, box: px })] }), cell([], W2, { paras: [imgPara(s.ihcImage, px, { before: 0, after: 0, box: px })] })],
  ]);
}
function nucleiStrip(s, px) {
  if (!s.nucleiImage) return [para([text("No nuclei field image is stored for this run.", { italics: true, size: 17, color: "595959" })])];
  return [
    imgPara(s.nucleiImage, px, { before: 40, after: 20 }),
    para([text("left: the IHC field as scanned   ·   middle: the blue counterstain the detector reads   ·   right: nuclei it found (green)", { size: 15, color: "595959" })], { align: AlignmentType.CENTER, after: 80 }),
  ];
}
const good = slide("CAN_00270", "A", "pixel"), bad1 = slide("CAN_00267", "A", "pixel"), bad2 = slide("CAN_00865", "A", "pixel");
const goodNuc = slides.filter((s) => s.nucleiImage && s.densityShortfall != null && s.densityShortfall >= 0).sort((a, b) => a.densityShortfall - b.densityShortfall)[0];
const evidence = [
  h1("Registration: what good and failed look like", { pageBreak: true, before: 0 }),
  h2(`Good — CAN_00270 · CD44 (image agreement ${good.nmi.toFixed(3)}, IHC tissue ${Math.round(good.tissueRatio * 100)}% of H&E)`, { before: 60 }),
  para("The outline lands on the same tumour nests on both slides: the large nodule at the bottom and the scattered foci at the top right keep their shape and position.", { after: 60 }),
  pairImages(good, 300),
  h2(`Failed — CAN_00267 · CD44 (image agreement ${bad1.nmi.toFixed(3)}, IHC tissue ${Math.round(bad1.tissueRatio * 100)}% of H&E)`, { pageBreak: true, before: 0 }),
  para([text("Two separate reasons: the image agreement is below the 1.008 floor, meaning the two images share almost no structure the registration could lock onto, and the IHC section holds about a third of the H&E's tissue — so part of the H&E has no counterpart to land on. The outline itself is only "), B(`${bad1.carriedMm2.toFixed(2)} mm²`), text(` here, and ${bad1.cells} tumour cell was measured.`)], { after: 60 }),
  pairImages(bad1, 300),
  h2(`Failed — CAN_00865 · CD44 (IHC tissue ${Math.round(bad2.tissueRatio * 100)}% of H&E)`, { pageBreak: true, before: 0 }),
  para("The IHC section holds only about a fifth of the H&E's tissue — most of it is missing or was not detected as tissue — so most of the outline has nothing real to land on. The image agreement passes, which shows why agreement alone is not enough.", { after: 60 }),
  pairImages(bad2, 300),
  h1("Nuclei segmentation: why cells go missing", { pageBreak: true, before: 0 }),
  para("Each strip is one field from the largest outlined region. The detector does not look at the brown stain at all; it reads the blue counterstain (middle image), because brown would otherwise be mistaken for nuclei. Where the brown is heavy, the blue underneath it is masked and the nucleus disappears from the middle image."),
  h2(`CAN_00270 · CD44 — ${Math.round(good.densityShortfall * 100)}% of nuclei missed`),
  para("The strongly stained tumour nest at the bottom almost vanishes from the counterstain channel, so the detected nuclei (green) sit mostly in the lightly stained stroma at the top. The cells that are measured are not the cells the score is about.", { after: 40 }),
  ...nucleiStrip(good, 640),
  h2(`CAN_00267 · CD44 — ${Math.round(bad1.densityShortfall * 100)}% of nuclei missed`),
  para("This is the failed registration above: the field is almost empty — background and a few strands of stroma — because the outline landed where there is no tumour.", { after: 40 }),
  ...nucleiStrip(bad1, 640),
  h2(`For contrast — ${goodNuc.case} · ${goodNuc.markerName} (${goodNuc.mode === "pixel" ? "pixel-wise" : "tile-based"}): ${Math.round(goodNuc.densityShortfall * 100)}% missed`),
  para("The lowest shortfall of the 60 runs: where the counterstain survives, the detector finds the nuclei.", { after: 40 }),
  ...nucleiStrip(goodNuc, 640),
  ...figure("fig_shortfall.png", "Figure 4. Share of nuclei missing on each IHC slide against the same regions on the H&E. Darker is worse. \"—\": not recorded for that run. \"more\": the IHC found more nuclei than the H&E."),
];

// ================================================================================================
// PART 4 - ONE PAGE PER CASE AND MARKER
// ================================================================================================
function slidePage(c, m, first) {
  const r = reader(c, m), p = slide(c, m, "pixel"), t = slide(c, m, "tile");
  const inR = (v, lo, hi) => v >= lo && v <= hi;
  const S = [1500, 800, 800, 800, 800, 1500, 2053, 2053];
  const aiCell = (v, lo, hi, mean, w, fmt, d, lab = "") => cell([text(fmt(v) + lab, { size: 18, bold: true }),
    text(`  ${sgn(v - mean, d)}${inR(v, lo, hi) ? "  ✓ in range" : ""}`, { size: 14, color: "595959" })], w, { fill: inR(v, lo, hi) ? "EAF4EA" : undefined });
  const scores = table(S, [[cell("", S[0], { fill: HEAD }), ...READERS.map((k, i) => cell(k, S[i + 1], { bold: true, fill: HEAD })), cell("Pathologists' mean (range)", S[5], { bold: true, fill: HEAD }),
    cell("AI pixel-wise", S[6], { bold: true, fill: HEAD }), cell("AI tile-based", S[7], { bold: true, fill: HEAD })]], [
    [cell("Percent positive", S[0], { bold: true, align: AlignmentType.LEFT }), ...READERS.map((k, i) => cell(`${n0(r.readers[k][0])}%`, S[i + 1])),
     cell(`${n0(Math.round(r.readerPctMean * 100) / 100)}%  (${n0(r.readerPctMin)}–${n0(r.readerPctMax)})`, S[5], { bold: true }),
     aiCell(p.percent, r.readerPctMin, r.readerPctMax, r.readerPctMean, S[6], n0, 0, "%"), aiCell(t.percent, r.readerPctMin, r.readerPctMax, r.readerPctMean, S[7], n0, 0, "%")],
    [cell("Intensity (0–2)", S[0], { bold: true, align: AlignmentType.LEFT }), ...READERS.map((k, i) => cell(n2(r.readers[k][1]), S[i + 1])),
     cell(`${n2(r.readerIntMean)}  (${n2(r.readerIntMin)}–${n2(r.readerIntMax)})`, S[5], { bold: true }),
     aiCell(p.intensity, r.readerIntMin, r.readerIntMax, r.readerIntMean, S[6], n2, 2), aiCell(t.intensity, r.readerIntMin, r.readerIntMax, r.readerIntMean, S[7], n2, 2)],
  ]);
  const D = [2700, 3803, 3803];
  const reg = (s) => `${failedCheck(s) ? "FAILED check" : "passed"} · agreement ${s.nmi.toFixed(3)} · IHC tissue ${Math.round((s.tissueRatio || 0) * 100)}% of H&E`;
  const nucl = (s) => s.densityShortfall == null ? "not recorded" : s.densityShortfall < 0 ? `IHC found ${Math.round(-s.densityShortfall * 100)}% more than H&E` : `${Math.round(s.densityShortfall * 100)}% missed vs H&E (${Math.round(s.densityPerMm2)} vs ${Math.round(s.heDensityPerMm2)} /mm²)`;
  const detail = table(D, [[cell("", D[0], { fill: SUB }), cell("Pixel-wise", D[1], { bold: true, fill: SUB }), cell("Tile-based", D[2], { bold: true, fill: SUB })]], [
    ["Tumour cells measured (positive)", (s) => `${s.cells.toLocaleString()} (${s.positive.toLocaleString()})`],
    ["Nuclei on the IHC slide", nucl],
    ["Outlined tumour", (s) => `${s.regions} region${s.regions === 1 ? "" : "s"} · ${s.carriedMm2.toFixed(2)} mm² · ${Math.round((s.coverage || 0) * 100)}% of invasive tumour`],
    ["Registration", reg],
    ["Raw values before rounding", (s) => `${s.percentRaw.toFixed(1)}% · brown density ${s.intensityRawOd.toFixed(3)} OD · H-score ${s.hScore.toFixed(0)}`],
  ].map(([lab, f]) => [cell(lab, D[0], { align: AlignmentType.LEFT, bold: true, size: 16 }), cell(f(p), D[1], { size: 16 }), cell(f(t), D[2], { size: 16, })]));
  // warnings, merged across the two methods
  const merged = new Map();
  for (const [s, tag] of [[p, "pixel-wise"], [t, "tile-based"]]) for (const w of warnings(s)) {
    const k = w.label; if (!merged.has(k)) merged.set(k, { tags: [], detail: w.detail }); merged.get(k).tags.push(tag);
  }
  const warnParas = merged.size === 0 ? [para([text("No slide-specific warnings.", { size: 16 })], { after: 0 })] :
    [...merged.entries()].map(([k, v]) => para([text(`${k} (${v.tags.join(", ")}): `, { bold: true, size: 15 }), text(v.detail, { size: 15 })], { after: 20 }));
  const W4 = CONTENT / 4, IMG = 158;
  const box = (f) => cell([], W4, { paras: [imgPara(f, IMG, { before: 0, after: 0, box: IMG })], pad: 20 });
  const images = table([W4, W4, W4, W4], [], [
    [cell([B("Pixel-wise"), text(" — H&E", { size: 15 })], W4, { fill: SUB, size: 15 }), cell([B("Pixel-wise"), text(" — IHC", { size: 15 })], W4, { fill: SUB, size: 15 }),
     cell([B("Tile-based"), text(" — H&E", { size: 15 })], W4, { fill: SUB, size: 15 }), cell([B("Tile-based"), text(" — IHC", { size: 15 })], W4, { fill: SUB, size: 15 })],
    [box(p.heImage), box(p.ihcImage), box(t.heImage), box(t.ihcImage)],
  ]);
  const out = [];
  if (first) out.push(h1(`Case ${c}`, { pageBreak: true, before: 0 }));
  out.push(h2(`${c}  ·  ${NAME[m]}  (marker ${m}, ${COMPARTMENT[m]} stain)`, { pageBreak: !first, before: first ? 40 : 0 }));
  out.push(scores, para("", { after: 60 }), detail, para("", { after: 50 }));
  out.push(table([CONTENT], [], [[cell([], CONTENT, { paras: [para([text("Warnings", { bold: true, size: 16 })], { after: 20 }), ...warnParas], fill: WARN, align: AlignmentType.LEFT })]]));
  out.push(para("", { after: 50 }), images);
  out.push(para([text("Nuclei found on the IHC slide (pixel-wise outline, largest region):", { bold: true, size: 15 })], { before: 60, after: 0, keepNext: true }));
  out.push(...nucleiStrip(p, 450));
  return out;
}
const perSlide = [
  h1("Slide-by-slide review", { pageBreak: true, before: 0 }),
  para("One page per case and marker, so each slide can be judged without opening it. Each page shows:"),
  bullet([B("Scores: "), text("all four pathologists, their mean and range, and both AI scores with their difference from the mean. A green cell is inside the pathologists' range.")]),
  bullet([B("Measurement details: "), text("how many tumour cells were measured, how many nuclei the IHC slide lost against the H&E, how much tumour was outlined, and whether the registration passed its check.")]),
  bullet([B("Warnings: "), text("the pipeline's own warnings for that slide, in plain words. The two that apply to every slide — provisional cut points and machine-confirmed registration — are left out here and explained under \"Is it reliable?\".")]),
  bullet([B("Images: "), text("the H&E and IHC outlines for both methods, and one field of the nuclei the detector found on the IHC slide.")]),
  ...CASES.flatMap((c) => MARKERS.flatMap((m, k) => slidePage(c, m, k === 0))),
];

// ================================================================================================
const nextSteps = [
  h1("What to do next", { pageBreak: true, before: 0 }),
  numbered([B("Fix nucleus detection on heavily stained IHC first. "), text("It affects nearly every slide and no calibration can recover cells that were never found. Options: detect nuclei on the H&E and carry them across, or use a detector trained on DAB-stained tissue.")], L3),
  numbered([B("Adopt refitted cut points. "), text("The fit exists and is tested on held-out cases; it clearly improves intensity. calibrate_marker_cuts.py can write it once it is pointed at the stored runs used here.")], L3),
  numbered([B("Have a person confirm every registration, "), text("starting with CAN_00267 and CAN_00865 CD44, and decide whether a failed check should drop the marker rather than score it.")], L3),
  numbered([B("Ask OncoStem whether 80% is a reporting ceiling "), text("for N-cadherin and Pan-cadherin, and score those markers the same way if it is.")], L3),
  numbered([B("Then re-score and repeat this comparison, "), text("keeping at least one case out of any fitting.")], L3),
];

// "Things that need attention" - appended to build.js before the Document is assembled.
const at = J("attention.json"), stale = at.stale;
const A = at.agreement, W8 = at.weighting, SEN = at.sensitivity, PVT = at.pixelVsTile;
const f270p = at.funnel["CAN_00270:A:pixel"], f270t = at.funnel["CAN_00270:A:tile"];
const red = (t) => text(t, { bold: true, color: "B42318" }), amber = (t) => text(t, { bold: true, color: "B54708" }), gold = (t) => text(t, { bold: true, color: "8A6D00" });
const need = (items) => [para([B("Needs checking:")], { after: 30, keepNext: true }), ...items.map((t) => bullet([text(t)], { after: 40 }))];
const h3 = (t) => new Paragraph({ keepNext: true, spacing: { before: 160, after: 70 }, children: [text(t, { size: 22, bold: true, color: "1F3864" })] });
const evidencePair = (fileL, fileR, capL, capR, px = 300) => {
  const W2 = CONTENT / 2;
  return table([W2, W2], [], [
    [cell([B(capL)], W2, { fill: SUB, size: 16, keepNext: true }), cell([B(capR)], W2, { fill: SUB, size: 16, keepNext: true })],
    [cell([], W2, { paras: [imgPara(fileL, px, { before: 0, after: 0, box: px })] }), cell([], W2, { paras: [imgPara(fileR, px, { before: 0, after: 0, box: px })] })],
  ]);
};
const INK = "0B0B0B";
const statusCell = (s, w) => cell([text(s, { bold: true, size: 16, color: { Verified: "1E7B34", Partly: "B54708", "Not tested": "52514E", Contradicted: "B42318" }[s] || INK })], w);

const s865p = slide("CAN_00865", "A", "pixel"), s865t = slide("CAN_00865", "A", "tile");
const s270t = slide("CAN_00270", "A", "tile");
const selRows = CASES.map((c) => at.selection[`${c}:pixel`]).filter(Boolean);

const attention = [
  h1("Things that need attention", { pageBreak: true, before: 0 }),
  para([B("Bottom line. "), text("The pipeline produces its numbers mechanically and correctly — every percent re-derives from the stored cells. What is not yet established is whether each transformation measures the biological quantity OncoStem's pathologists report. The riskiest links are "), B("H&E tumour outline → IHC registration → the right cells counted"), text(", and then "), B("right cells → right stain measurement → the right clinical percentage"), text(". Every item below is measured on this cohort's stored runs; where something could not be measured here it says so.")]),
  para([red("Red"), text(" = can invalidate a score today.   "), amber("Amber"), text(" = a consequential assumption that is untested.   "), gold("Yellow"), text(" = robustness to establish before scaling up.")], { after: 160 }),

  // ---------------------------------------------------------------- 1. critical
  h2("1. Critical concerns", { before: 60 }),
  h3("A. Tumour cells are not reliably identified — the cell-typing step fails its own check on every run"),
  para([red("Found in this analysis. "), text(`Step 14 separates tumour cells from stromal and immune cells by nucleus size and shape, and refuses to call itself trustworthy when the typical nucleus is smaller than 25 µm² — the floor for a real epithelial nucleus. On all 101 stored runs it reported "not trustworthy": the median segmented nucleus is about 14 µm², against about 43–45 µm² on the same case's H&E. The rules are therefore sorting fragments of nuclei, not cells, and "tumour cells" in every score is an estimate built on fragments. The tumour cells counted are ${Math.round(at.tumourShareOfNuclei.pixel.median * 100)}% of the nuclei detected (median, pixel-wise; range ${Math.round(at.tumourShareOfNuclei.pixel.min * 100)}–${Math.round(at.tumourShareOfNuclei.pixel.max * 100)}%).`)]),
  ...need(["tumour-cell precision, recall and F1 against a pathologist's cell labels", "a confusion matrix: tumour / immune / stromal / other", "whether classification changes with stain intensity — if strongly stained cells are classed differently, the percent is biased", "fix segmentation first: size rules cannot work on fragments"]),

  h3("B. Nuclei are missed on the IHC slide"),
  para(`The IHC section yields ${Math.round(avgShort("pixel") * 100)}% (pixel-wise) and ${Math.round(avgShort("tile") * 100)}% (tile-based) fewer nuclei per mm² than the same regions on the H&E (Figure 4). In the fields inspected (CAN_00270 CD44 below), the heavily stained tumour nest disappears from the counterstain channel and the detector finds nuclei mostly in the stroma — so the cells lost are the stained tumour cells. Across slides, though, the shortfall does not track how much of the tumour the pathologists called positive (Spearman ρ = ${at.shortfallVsStain.pixel} pixel-wise, ${at.shortfallVsStain.tile} tile-based; Figure 9, right), so the direction and size of the bias on the final percent are not established yet.`),
  ...nucleiStrip(good, 600),
  ...need(["nuclei precision, recall and F1 on IHC fields a person has annotated", "nucleus recall plotted against DAB intensity — the test for preferential loss of positive cells", "H&E-to-IHC nuclei density ratio as a hard QC gate, not a warning", "misses near tissue edges and in crowded, touching nuclei", "alternatives: detect on the H&E and carry nuclei across, or a detector trained on DAB-stained tissue"]),

  h3("C. The cut points are not calibrated"),
  para(`Both files are scored with provisional thresholds (0.15 brown density for membrane markers, 0.12 for cytoplasmic; ring completeness ≥ 0.35 or stained fraction ≥ 0.25). The reader sheet now exists; refitting on it, tested on held-out cases, moves percent from ${sb["pixel: as delivered"].pctMae.toFixed(1)} to ${sb["pixel: cut refit (held-out)"].pctMae.toFixed(1)} points pixel-wise and intensity inside the pathologists' range on ${sb["tile: cut refit (held-out)"].inInt} of 30 tile-based slides. ABCC4's fitted cut is about three times the provisional one. Calibration helps, but not enough on its own while A and B stand.`),
  ...need(["fit per marker on held-out cases (done in this document) and adopt it", "report MAE, bias, RMSE, correlation and Bland–Altman after the fit (Section 9)", "keep at least one case out of every fit"]),

  h3("D. Registration: working, but unreviewed — and two failures were let through"),
  para(`All 60 registrations were confirmed by the batch run; nobody has looked at them. The automated check uses image agreement (mutual information) and the tissue-area ratio. Two CD44 slides failed it and were scored anyway: CAN_00267 (agreement 1.006, below the 1.008 floor; IHC holds 36% of the H&E's tissue) and CAN_00865 (IHC holds 19%). Their scores should not be read as measurements. CAN_00865 also shows why agreement alone is not enough: it passed the agreement test while most of the section has no counterpart.`),
  evidencePair(bad2.heImage, bad2.ihcImage, "CAN_00865 CD44 — H&E outline", "IHC after registration: most of the section is missing", 280),
  ...need(["a person confirms every pair, starting with the two failures", "tissue overlap (Dice/IoU) and tumour-boundary overlap between the registered sections", "landmark error in µm on a few hand-placed points per pair", "fold count (0 on all 60 today) and local deformation", "near-blank slides validated separately", "a decision rule: should a failed check drop the marker instead of scoring it?"]),

  h3("E. The denominator shrinks at every stage"),
  para(`Each stage narrows what is measured. For CAN_00270 CD44: ${f270p.tissueMm2.toFixed(0)} mm² of tissue → ${f270p.invasiveMm2.toFixed(1)} mm² invasive → ${f270p.outlinedMm2.toFixed(1)} mm² pixel-wise outline (${f270t.outlinedMm2.toFixed(1)} tile-based) → ${f270p.sampledMm2.toFixed(1)} mm² sampled → ${f270p.nucleiCounted.toLocaleString()} nuclei → ${f270p.tumourCells.toLocaleString()} tumour cells → ${f270p.positive.toLocaleString()} positive. Tile-based, only ${f270t.sampledMm2.toFixed(1)} mm² of its ${f270t.outlinedMm2.toFixed(1)} mm² outline is sampled. Across the cohort the tile-based sampled area is a median ${Math.round(at.sampledShareOfOutlined.tile.median * 100)}% of the outline (as low as ${(at.sampledShareOfOutlined.tile.min * 100).toFixed(1)}%). Pixel-wise, the sampled field area exceeds the outline on 20 of 30 slides (median ${at.sampledShareOfOutlined.pixel.median.toFixed(2)}×), because each field is a fixed square larger than the small BEETLE regions.`),
  ...figure("fig_funnel.png", "Figure 5. The measurement funnel for CAN_00270 CD44: area at each stage, and the cells that remain."),
  ...need(["whether each exclusion is justified and unbiased", "that cells outside the outline are excluded when a sampling field is larger than the region it samples", "how many fields per region are needed for a stable percent (sample more, compare)"]),

  h3(stale.length ? `F. A data-lineage bug: stale files mixed into ${stale.length} score${stale.length === 1 ? "" : "s"}` : "F. Data lineage: region files match their reports"),
  para(stale.length
    ? `The per-cell folder behind ${stale.map((x) => `${x.case} ${x.markerName} (${x.mode === "pixel" ? "pixel-wise" : "tile-based"})`).join(", ")} holds region files left behind by an earlier run beside the ones the scoring run wrote, and the scoring read all of them. ` +
      stale.map((x) => `${x.case} ${x.markerName}: ${x.filesOnDisk} region files for ${x.regionsInReport} listed regions; the CSV says ${x.csvCells.toLocaleString()} cells, the correct count is ${x.correctCells.toLocaleString()}; the percent is ${x.csvPercent}% delivered and ${x.correctPercent}% recomputed on the listed regions.`).join(" ") +
      ` The other ${at.rowsChecked - stale.length} scores were checked and are clean.`
    : `All ${at.rowsChecked} scores were checked: every per-cell folder holds exactly the regions its report lists.`),
  ...need(["clear a pair's per-cell folder before a run writes into it", "assert that the region files on disk match the regions in the report before scoring"]),

  // ---------------------------------------------------------------- 2. region selection
  h2("2. Region selection (step 10)", { pageBreak: true, before: 0 }),
  para("Step 10 offers the invasive regions step 9 found and keeps a subset; the rest are dropped as too small or over the cap. In every case here the default selection was used — no person chose the regions."),
  table([1500, 1300, 1200, 1900, 1500, 1500, 1406], [[cell("Case", 1500, { bold: true, fill: HEAD }), cell("Candidates", 1300, { bold: true, fill: HEAD }), cell("Selected", 1200, { bold: true, fill: HEAD }),
    cell("Dropped as too small", 1900, { bold: true, fill: HEAD }), cell("Invasive (mm²)", 1500, { bold: true, fill: HEAD }), cell("Offered (mm²)", 1500, { bold: true, fill: HEAD }), cell("Chosen by", 1406, { bold: true, fill: HEAD })]],
    CASES.map((c) => { const s = at.selection[`${c}:pixel`]; return [cell(c, 1500), cell(String(s.candidates), 1300), cell(String(s.selected), 1200),
      cell(`${s.droppedSmall} (${s.droppedSmallMm2.toFixed(2)} mm²)`, 1900), cell(s.invasiveMm2.toFixed(1), 1500), cell(s.offeredMm2.toFixed(1), 1500), cell(s.chosenByPerson ? "a person" : "default", 1406)]; })),
  para([B("Assumptions to test. "), text("(1) The selected regions represent the whole invasive tumour. (2) Tiny regions can be dropped: on CAN_00270, 58 fragments totalling 2.9 mm² were. (3) The default — largest regions first — is acceptable. If larger regions are systematically more cellular or more positive, the score is biased.")], { before: 120 }),
  ...need(["score with the largest regions, a random set, all regions, and a stratified sample, and compare", "set the minimum region area or cell count from data, not by assumption"]),

  // ---------------------------------------------------------------- 3. BEETLE
  h2("3. BEETLE (pixel-wise) vs tile outline"),
  para(`The choice of outline is not cosmetic. Switching between them moves the score by 5 points or more on ${PVT.changedBy5} of 30 slides and by 20 or more on ${PVT.changedBy20}, from ${PVT.min} to +${PVT.max} points. Cell counts move just as much: CAN_00267's pixel-wise outline yields 1–73 cells per marker against 54–980 tile-based; on CAN_00865 it is the other way round.`),
  imgPara(path.join(HERE, "fig_pixel_vs_tile.png"), 330),
  caption("Figure 6. Each dot is one slide: AI pixel-wise against AI tile-based percent. Dots off the grey band disagree by more than 5 points."),
  evidencePair(s865p.heImage, s865t.heImage, "CAN_00865 CD44 — pixel-wise outline (BEETLE)", "the same slide — tile-based outline", 280),
  para([B("The assumption "), text("is that BEETLE's pixel boundary is closer to the true invasive boundary than the tiles. That has not been shown. Comparing tile against BEETLE only says they differ; it does not say which is right.")], { before: 100 }),
  ...need(["Dice, IoU, precision, recall and boundary distance of both outlines against a pathologist-drawn reference mask", "area error, and recall of small and large tumour regions separately", "tile → BEETLE → reference, not tile vs BEETLE"]),

  // ---------------------------------------------------------------- 4. weighting
  h2("4. How the percent is combined across regions", { pageBreak: true, before: 0 }),
  para(`Three ways to turn per-region counts into one number: weight each region by its area (delivered), pool every measured cell, or average the regions plainly. Across the cohort they land about equally far from the pathologists (Figure 7) — the formula is not the main source of error today. On single slides it matters a great deal: the three differ by 10 points or more on ${W8.pixel.spreadOver10} pixel-wise and ${W8.tile.spreadOver10} tile-based slides, up to ${W8.pixel.spreadMax} points. For CAN_00270 CD44 they give ${at.weighting270A.percent_area_weighted.toFixed(1)}%, ${at.weighting270A.percent_pooled.toFixed(1)}% and ${at.weighting270A.percent_plain_mean.toFixed(1)}%; the pathologists say 80%.`),
  imgPara(path.join(HERE, "fig_weighting.png"), 430),
  caption("Figure 7. Average difference from the pathologists for each way of combining regions."),
  para([B("The key question is what clinical quantity is being reproduced: "), text("the share of tumour cells positive, the share of tumour area positive, an average of field-level positivity, or the pathologist's visual estimate. These are not the same measurement. Decide it with OncoStem before fixing the formula — the reader sheet alone cannot tell them apart while A and B stand.")]),

  // ---------------------------------------------------------------- 5. cell rules
  h2("5. Cell-level rules"),
  para(`A membrane cell is positive when its shell averages at least 0.15 brown density and at least 35% of the ring is stained; a partly stained ring counts as a whole cell. Sweeping these on the stored cells (Figure 8): the cohort-wide error barely moves with ring completeness between 0.20 and 0.50, but single slides do — between those two settings a membrane slide's percent moves by a median ${SEN.completenessMove.pixel.median} points and up to ${SEN.completenessMove.pixel.max} (pixel-wise). One global density cut cannot fix the markers together: the best single cut sits near the provisional one, while the per-marker fits (Section "The refitted cut points") differ threefold.`),
  ...figure("fig_sensitivity.png", "Figure 8. Average difference from the pathologists as the ring-completeness rule (left) and the positivity cut (right) are varied."),
  para([B("Not testable here: "), text("the 1.5 µm membrane shell. The stored measurements were taken at one shell width; testing 1.0–4.0 µm needs a re-measurement of step 16, and whether one width suits all five markers is open.")]),
  ...need(["shell width 1.0 / 1.5 / 2.0 / 3.0 / 4.0 µm against expert-labelled cells", "ring completeness 0.20–0.50 against expert-labelled cells", "the partial-ring rule: count / half / exclude", "shell contamination from neighbouring cells in crowded tumour"]),

  // ---------------------------------------------------------------- 6-8
  h2("6. Stain measurement", { pageBreak: true, before: 0 }),
  para("Brown is measured as optical density after separating the stains with a fixed, published colour basis (Ruifrok), on each slide's own white point. A per-slide estimated basis (Macenko) is computed alongside for comparison; on CAN_00270 CD44 its counterstain direction differs from the fixed one by about 19°. Fixed vectors worked better for nucleus detection in an earlier comparison, but they have not been validated across the cohort."),
  ...need(["white-reference stability per slide", "saturation and very dark pixels in heavily stained tumour", "DAB/counterstain separation quality, marker by marker", "scanner, batch and case variation"]),
  h2("7. Intensity: several scales, weak agreement"),
  para(`The pipeline carries four scales: per-cell 0 / 1+ / 2+ / 3+, OncoStem's 0–2 bands (the delivered one), Allred and H-score. Against the pathologists on the 0–2 scale, the AI's band matches their nearest band exactly on ${A.pixel.intExact} (pixel-wise) and ${A.tile.intExact} (tile-based) of 30 slides, is within 0.5 on ${A.pixel.intWithin05} and ${A.tile.intWithin05}, and the quadratic-weighted kappa is ${A.pixel.intKappa} and ${A.tile.intKappa} — fair agreement at best. OncoStem grades intensity against internal control ducts on the same slide; an absolute band table is a stand-in for that relative judgement.`),
  ...need(["one documented mapping between the four scales", "exact and ±1-band agreement, confusion matrix and weighted kappa after calibration", "whether internal-control grading can be measured"]),
  h2("8. Low cell counts"),
  para(`${fl.pixel.tally["Too few cells to measure"]} pixel-wise results rest on fewer than 50 cells and more are under 400. Their average error is not dramatically worse (${at.errorByCells.pixel.under400.mae} points under 400 cells vs ${at.errorByCells.pixel.atLeast400.mae} above, pixel-wise) — only because larger errors elsewhere dominate. A result from 1–20 cells should never be reported as equivalent to one from 5,000.`),
  ...figure("fig_cells_shortfall.png", "Figure 9. Left: error against the number of cells measured. Right: nuclei missing against how much of the tumour the pathologists called positive."),
  ...need(["treat < 50 cells as no result, not a warning", "set the thin-count threshold from data: how the percent stabilises as cells are added"]),

  // ---------------------------------------------------------------- 9 agreement
  h2("9. Agreement with the pathologists — the full set", { pageBreak: true, before: 0 }),
  table([3806, 2200, 2200, 2100], [[cell("Measure (30 slides)", 3806, { bold: true, fill: HEAD }), cell("AI pixel-wise", 2200, { bold: true, fill: HEAD }), cell("AI tile-based", 2200, { bold: true, fill: HEAD }), cell("A pathologist vs the other three", 2100, { bold: true, fill: HEAD })]], [
    ["Percent: average difference (MAE)", A.pixel.mae, A.tile.mae, A.readerLeaveOneOut.mae],
    ["Percent: median difference", A.pixel.medianAe, A.tile.medianAe, "—"],
    ["Percent: RMSE", A.pixel.rmse, A.tile.rmse, "—"],
    ["Percent: bias (AI − pathologists)", sgn(A.pixel.bias), sgn(A.tile.bias), "—"],
    ["Percent: Pearson r / Spearman ρ", `${A.pixel.pearson} / ${A.pixel.spearman}`, `${A.tile.pearson} / ${A.tile.spearman}`, "—"],
    ["Percent: within ±5 points", `${A.pixel.within5}/30`, `${A.tile.within5}/30`, "—"],
    ["Percent: within ±10 points", `${A.pixel.within10}/30`, `${A.tile.within10}/30`, `${A.readerLeaveOneOut.within10}/${A.readerLeaveOneOut.n}`],
    ["Percent: within ±20 points", `${A.pixel.within20}/30`, `${A.tile.within20}/30`, "—"],
    ["Percent: Bland–Altman 95% limits", `${ag.pixel.baLow} to +${ag.pixel.baHigh}`, `${ag.tile.baLow} to +${ag.tile.baHigh}`, "—"],
    ["Intensity: exact band", `${A.pixel.intExact}/30`, `${A.tile.intExact}/30`, "—"],
    ["Intensity: within 0.5", `${A.pixel.intWithin05}/30`, `${A.tile.intWithin05}/30`, "—"],
    ["Intensity: weighted kappa", A.pixel.intKappa, A.tile.intKappa, "—"],
  ].map(([k, a, b, c]) => [cell(k, 3806, { align: AlignmentType.LEFT }), cell(String(a), 2200, { bold: true }), cell(String(b), 2200, { bold: true }), cell(String(c), 2100)])),
  para("A single pathologist lands within 10 points of the other three on nearly every slide; the AI does on about a quarter. Clinical-category agreement (sensitivity, specificity, PPV, NPV) needs OncoStem's category cut-offs, which are not in the reader sheet.", { before: 100 }),

  // ---------------------------------------------------------------- 10 checklist
  h2("10. What to measure, stage by stage"),
  ...[
    ["Slide QC", [["Tissue detection", "Dice / IoU against a reference mask", "Not tested"], ["Artefact detection", "sensitivity / specificity", "Not tested"], ["White reference", "reproducibility per slide", "Not tested"]]],
    ["Tumour detection", [["Invasive vs non-invasive", "sensitivity / specificity", "Not tested"], ["Invasive segmentation", "Dice, IoU, boundary distance", "Not tested"], ["Small / large regions", "recall / precision", "Not tested"]]],
    ["BEETLE refinement", [["Outline vs reference", "Dice, IoU, precision, recall, area error", "Not tested"], ["Tile vs BEETLE", "score change per slide", "Measured: 19/30 move ≥ 5 points"]]],
    ["Registration", [["Pass / fail", "automated check", "Measured: 2/30 fail"], ["Folds", "fold count, Jacobian", "Measured: 0 folds"], ["Anatomical match", "tissue & tumour Dice, landmark error, visual review", "Not tested"]]],
    ["Nucleus segmentation", [["Completeness", "IHC vs H&E nuclei density", "Measured: 68–76% missing"], ["Accuracy", "precision, recall, F1 on annotated fields", "Not tested"], ["Stain bias", "recall vs DAB intensity", "Not tested (visual evidence only)"]]],
    ["Tumour-cell typing", [["Trust gate", "median nucleus area ≥ 25 µm²", "Measured: fails on all runs"], ["Accuracy", "precision, recall, confusion matrix", "Not tested"]]],
    ["Cell measurement", [["Shell width", "1.0–4.0 µm sweep", "Not tested"], ["Ring completeness", "0.20–0.50 sweep", "Measured (Figure 8)"]]],
    ["Final score", [["Combining regions", "pooled / area / plain vs pathologists", "Measured (Figure 7)"], ["Agreement", "MAE, RMSE, Bland–Altman, kappa", "Measured (Section 9)"]]],
  ].flatMap(([stage, items]) => [h3(stage), table([2800, 4500, 3006], [[cell("Check", 2800, { bold: true, fill: HEAD }), cell("How to measure", 4500, { bold: true, fill: HEAD }), cell("Status here", 3006, { bold: true, fill: HEAD })]],
    items.map(([a, b, c]) => [cell(a, 2800, { align: AlignmentType.LEFT }), cell(b, 4500, { align: AlignmentType.LEFT }), cell(c, 3006, { align: AlignmentType.LEFT, bold: c.startsWith("Measured") })]))]),

  // ---------------------------------------------------------------- 11 assumptions
  h2("11. Assumptions register", { pageBreak: true, before: 0 }),
  para("Every assumption the scores rest on, with what this cohort's data says about it."),
  table([500, 4600, 1500, 3706], [[cell("#", 500, { bold: true, fill: HEAD }), cell("Assumption", 4600, { bold: true, fill: HEAD }), cell("Status", 1500, { bold: true, fill: HEAD }), cell("Evidence here", 3706, { bold: true, fill: HEAD })]],
    [
      ["H&E and IHC sections hold corresponding tissue", "Partly", "true on 28/30 pairs by tissue ratio; not on CAN_00267 and CAN_00865 CD44"],
      ["The H&E invasive outline can be carried to the IHC", "Partly", "visually on the good pairs; 2 failures"],
      ["The registration is anatomically plausible", "Partly", "0 folds on all 60; no landmark test"],
      ["Image agreement (mutual information) is enough to judge a registration", "Contradicted", "CAN_00865 passes agreement with 19% of the tissue"],
      ["BEETLE's pixel boundary is the true invasive boundary", "Not tested", "differs from tiles by −55 to +65 points"],
      ["Tiny regions can be dropped", "Not tested", "up to 58 fragments (2.9 mm²) per case"],
      ["The selected regions represent the tumour", "Not tested", "all default selections"],
      ["Fixed stain vectors hold across slides", "Not tested", "per-slide estimate drifts ~19° on one slide"],
      ["Optical density is comparable across slides", "Not tested", ""],
      ["Nucleus segmentation is complete enough", "Contradicted", "68–76% of nuclei missing on average"],
      ["Missed nuclei are not systematically positive", "Partly", "visually violated in CAN_00270 fields; no slide-level correlation"],
      ["Tumour-cell typing is accurate", "Contradicted", "fails its own trust gate on all 101 runs"],
      ["A 1.5 µm shell is the right membrane compartment", "Not tested", "needs re-measurement"],
      ["Ring completeness ≥ 0.35 marks a positive membrane", "Partly", "cohort error flat 0.20–0.50; single slides move up to 21 points"],
      ["0.15 / 0.12 density is the right positivity cut", "Contradicted", "per-marker fits differ up to threefold"],
      ["The intensity bands are right", "Contradicted", "exact band on 7–11/30; kappa ≈ 0.3"],
      ["Area weighting is the intended clinical measurement", "Not tested", "formulas differ up to 58 points on one slide"],
      ["Low-cell results can be identified and excluded", "Partly", "flagged, but still reported as scores"],
      ["One rule set suits all five markers", "Contradicted", "fitted cuts differ by marker"],
      ["The pipeline agrees with the pathologists", "Contradicted", "within ±10 points on 7/30 slides"],
    ].map(([a, s, e], i) => [cell(String(i + 1), 500), cell(a, 4600, { align: AlignmentType.LEFT }), statusCell(s, 1500), cell(e, 3706, { align: AlignmentType.LEFT, size: 16 })])),

  // ---------------------------------------------------------------- 12 priority
  h2("12. Priority order"),
  para([red("1 · Nucleus segmentation and cell typing on IHC. "), text("Every score is built on them, and both fail their own checks today. No calibration can recover cells that were never found.")], { after: 70 }),
  para([red("2 · Pathologist calibration. "), text("Fit the cut points per marker on held-out cases, then re-measure agreement.")], { after: 70 }),
  para([red("3 · Registration review. "), text("A person confirms all 60 pairs; resolve or exclude the two failures.")], { after: 70 }),
  para([red("4 · BEETLE vs reference outline. "), text("Show that the pixel boundary is more accurate than the tiles rather than assume it.")], { after: 70 }),
  para([red("5 · Define the clinical quantity. "), text("Cell-based, area-based or field-based percent — agree it with OncoStem, then fix the formula.")], { after: 70 }),
  para([amber("6 · Cell-level rules. "), text("Density cut, ring completeness, shell width, partial-ring treatment — against expert-labelled cells.")], { after: 70 }),
  para([amber("7 · Low-denominator rules. "), text("Evidence-based minimum cells and area; below it, no score.")], { after: 70 }),
  para([amber("8 · Data lineage. "), text("Clear per-cell folders before each run and assert region files match the report.")], { after: 70 }),
  para([gold("9 · Robustness. "), text("Scanner, stain batch, case and marker variation, once the above are in place.")], { after: 70 }),
];

const doc = new Document({
  creator: "Cancer Scoring System",
  title: "AI Scores vs Pathologists",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 240, after: 110 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: FONT, color: "2E5597" }, paragraph: { spacing: { before: 200, after: 90 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [text("AI Scores vs Pathologists  ·  page ", { size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "808080" })] })] }) },
    children: [...title, ...glance, ...how, ...whyZero, ...reliability, ...graphs, ...calib, ...evidence, ...nextSteps, ...perSlide, ...attention],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, (buf.length / 1e6).toFixed(1), "MB"); });
