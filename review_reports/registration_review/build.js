// Builds the pathologist review document for the H&E -> IHC registrations.
// One page per case and marker: pixel-wise H&E -> IHC on top, tile-based H&E -> IHC below.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, HeadingLevel, BorderStyle, ShadingType,
  Footer, PageNumber, LevelFormat, VerticalAlign,
} = require("docx");

// Reads what the Python steps wrote under v<N>_data/data/review_reports/registration_review.
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
const HERE = path.join(ROOT, "storage", `${dataVersion()}_data`, "data", "review_reports", "registration_review");
const rows = JSON.parse(fs.readFileSync(path.join(HERE, "rows.json"), "utf8"));
const OUT = process.argv[2];

// A4 portrait, 1.5 cm margins.
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 850;
const CONTENT = PAGE_W - 2 * MARGIN; // 10206
const HALF = CONTENT / 2;
const IMG_PX = 335; // ~8.9 cm at 96 dpi; two rows of two fit one page with their details

const FONT = "Calibri";
const GREY = "F2F2F2", BLUE = "DCE6F2", RULE = "BFBFBF";
const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const borders = { top: border, bottom: border, left: border, right: border };
const MODE_LABEL = {
  pixel: "Pixel-wise registration (BEETLE outline)",
  tile: "Tile-based registration (step 9 squares)",
};

const f1 = (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const change = (he, ihc) => {
  if (!he) return "—";
  const s = (((ihc - he) / he) * 100).toFixed(1);
  return s.startsWith("-") ? `${s}%` : `+${s}%`;
};

function text(t, opts = {}) {
  return new TextRun({ text: t, font: FONT, size: opts.size || 20, bold: opts.bold, italics: opts.italics, color: opts.color });
}
function para(children, opts = {}) {
  return new Paragraph({
    children: Array.isArray(children) ? children : [text(children, opts)],
    alignment: opts.align, spacing: { before: opts.before ?? 0, after: opts.after ?? 80 },
    keepNext: opts.keepNext, keepLines: true, numbering: opts.numbering,
  });
}
function bullet(children) {
  return para(children, { numbering: { reference: "bullets", level: 0 } });
}
function cell(children, width, opts = {}) {
  return new TableCell({
    children: Array.isArray(children) ? children : [children],
    width: { size: width, type: WidthType.DXA }, borders,
    columnSpan: opts.span, verticalAlign: VerticalAlign.CENTER,
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
  });
}
function image(file) {
  return new ImageRun({ type: "jpg", data: fs.readFileSync(file), transformation: { width: IMG_PX, height: IMG_PX },
    altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } });
}

// --- one method's H&E -> IHC pair, with its areas --------------------------------------
function pair(r) {
  const detail = (label, value) => [text(`${label}: `, { bold: true, size: 17 }), text(value + "    ", { size: 17 })];
  const row = (cells) => new TableRow({ cantSplit: true, children: cells });
  const keep = { after: 0, keepNext: true };
  return new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [HALF, HALF],
    rows: [
      row([cell(para([text(MODE_LABEL[r.mode], { bold: true, size: 21 })], keep), CONTENT, { span: 2, fill: BLUE })]),
      row([
        cell(para([text(`H&E with invasive segmentation — ${r.heFile}`, { bold: true, size: 16 })], { ...keep, align: AlignmentType.CENTER }), HALF, { fill: GREY }),
        cell(para([text(`IHC with registered segmentation — ${r.ihcFile}`, { bold: true, size: 16 })], { ...keep, align: AlignmentType.CENTER }), HALF, { fill: GREY }),
      ]),
      row([
        cell(para([image(r.heImage)], { ...keep, align: AlignmentType.CENTER }), HALF),
        cell(para([image(r.ihcImage)], { ...keep, align: AlignmentType.CENTER }), HALF),
      ]),
      row([cell([
        para([...detail("Regions outlined", String(r.regions)),
              ...detail("Outlined area on H&E", `${f1(r.heRegionMm2, 2)} mm²`),
              ...detail("Same outline on IHC", `${f1(r.ihcRegionMm2, 2)} mm² (${change(r.heRegionMm2, r.ihcRegionMm2)})`)], { after: 30, keepNext: true }),
        para([...detail("Invasive tumour called on whole H&E", `${f1(r.invasiveMm2, 2)} mm²`),
              ...detail("Share outlined", pct(r.coverage)),
              ...detail("Tissue area H&E / IHC", `${f1(r.heTissueMm2, 0)} / ${f1(r.ihcTissueMm2, 0)} mm²`)], { after: 0 }),
      ], CONTENT, { span: 2 })]),
    ],
  });
}

// --- front matter --------------------------------------------------------------------------
const cases = [...new Set(rows.map((r) => r.case))];
const markers = [...new Map(rows.map((r) => [r.marker, r.markerName])).entries()];
const h1 = (t, opts = {}) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: opts.pageBreak,
  children: [text(t, { size: 30, bold: true })], spacing: { before: opts.before ?? 200, after: 100 }, keepNext: true });

const intro = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [text("H&E → IHC Registration Review", { size: 48, bold: true })], spacing: { after: 120 } }),
  para([text(`${cases.length} cases · ${markers.length} markers · pixel-wise and tile-based · ${rows.length} registrations`, { size: 24, color: "404040" })], { after: 60 }),
  para([text(`Generated ${new Date().toISOString().slice(0, 10)} from the current pipeline results.`, { size: 20, italics: true, color: "595959" })], { after: 240 }),

  h1("Purpose", { before: 120 }),
  para("The score for each marker is measured only inside the invasive tumour outlined on the H&E slide. That outline is drawn on the H&E, then carried onto the matching IHC section by image registration. This document asks one question for every registration: does the red outline on the IHC land on the same tumour it marks on the H&E?"),

  h1("How each page is laid out"),
  para("One page per case and marker. Each page has two rows:"),
  bullet([text("Top row — pixel-wise: ", { bold: true }), text("H&E with the invasive outline (left) → the IHC slide with the same outline after registration (right).")]),
  bullet([text("Bottom row — tile-based: ", { bold: true }), text("the same H&E → IHC pair, with the tile outline carried across instead.")]),
  para("Under each row are the outlined area on both slides and the tissue area of each section.", { before: 60, after: 160 }),

  h1("The two methods"),
  para([text("Pixel-wise (BEETLE outline). ", { bold: true }), text("The invasive boundary is traced pixel by pixel inside the regions a person selected. It follows the tumour closely but usually covers less of it.")]),
  para([text("Tile-based (step 9 squares). ", { bold: true }), text("The outline is built from square tiles the tissue classifier called invasive. It covers more tumour but its edges are blocky and can include some surrounding tissue.")], { after: 100 }),
  para("Both rows on a page use the same registration, so the two IHC images differ only in which outline is carried across."),

  h1("What the numbers mean"),
  ...[
    ["Outlined area on H&E", "the total area inside the red outlines on the H&E."],
    ["Same outline on IHC", "that outline's area after registration, and the change from the H&E. The registration only moves, turns and uniformly rescales the section, so a large change means the two sections are cut or mounted at different sizes."],
    ["Invasive tumour called on whole H&E", "all invasive carcinoma the tissue classifier found on the H&E. Share outlined is the outlined area as a fraction of it."],
    ["Tissue area H&E / IHC", "total tissue on each section. A big mismatch can mean torn or missing tissue on one slide. For some cases the two rows show different tissue areas because the two runs used different tissue-detection thresholds; this affects only this figure, not the outlines or the registration."],
  ].map(([k, v]) => bullet([text(`${k}: `, { bold: true }), text(v)])),
];

// --- one page per case and marker -------------------------------------------------------
const body = [];
for (const c of cases) {
  markers.forEach(([marker, name], k) => {
    const pixel = rows.find((r) => r.case === c && r.marker === marker && r.mode === "pixel");
    const tile = rows.find((r) => r.case === c && r.marker === marker && r.mode === "tile");
    if (k === 0) body.push(h1(`Case ${c}`, { pageBreak: true, before: 0 }));
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, pageBreakBefore: k > 0, keepNext: true,
      children: [text(`${c}  ·  ${name} (marker ${marker})`, { size: 26, bold: true })], spacing: { before: 60, after: 100 } }));
    body.push(pair(pixel));
    body.push(para("", { after: 120, keepNext: true }));
    body.push(pair(tile));
  });
}

const doc = new Document({
  creator: "Cancer Scoring System",
  title: "H&E to IHC Registration Review",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT, color: "2E5597" }, paragraph: { spacing: { before: 120, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [text("H&E → IHC Registration Review  ·  page ", { size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "808080" })] })] }) },
    children: [...intro, ...body],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, (buf.length / 1e6).toFixed(1), "MB"); });
