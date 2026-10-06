"""Catalogue of the 19 pipeline steps.

Each entry records what a step does, why it must run at that point in the order,
how it would be built, whether it is implemented yet, and - `runs_on` - which
slide of a case it reads. All nineteen run for real.

**The pipeline is not a line, and `runs_on` is where that stops being a secret.**
Steps 1 to 6 are a per-slide *prefix*: they run once on the H&E and once on the
immunostained slide, because every one of them measures a property of one piece
of glass. A white point is the clearest case - the guide's word is that it "must
be per slide, that is the entire point" - but the same is true of the artefact
map, the tissue mask, the density and the unmix above them. Steps 7 to 11 then run
on the H&E alone, because that is where structure is legible and where both region
models were fitted; step 12 carries the refined boundary across; steps 13 to 18
measure on the immunostained slide, because that is where the brown is. Step 19 is
keyed on the case rather than on a slide.

Until this field existed, the second half of that prefix ran anyway - steps 3 and
4 were computed on the IHC slide *inside* steps 11 and 14 - but nothing said so
and no screen showed it, so a reader watching the demo saw an H&E tile captioned
as the picture the score was measured from. Declaring the role here, once, is what
lets the UI put the right slide under the right step instead of each panel
guessing.

Step 19 runs but reports that it has no pathologist readings to compare against,
which is a result rather than a gap - see `step19_validation/agreement.py`.

**Steps 10 and 11 are where BEETLE stopped being a whole-slide alternative.** It used
to be one of step 7's three options, and choosing it meant segmenting the entire section
per pixel. It is now the second half of a coarse-to-fine pair: step 8's tile head finds
candidate regions, a person picks which of them matter, and BEETLE runs on those alone.
What comes out of step 11 is the invasive-tumour mask every step after it measures
inside - step 12 refuses to carry the coarse tile regions across in its place.

Source of truth for the wording: docs/demo-pipeline-guide.md
"""

from app.schemas.pipeline import Approach, Branch, PipelineStage, SlideRole

#: Steps 1-6, the per-slide prefix: each runs once per slide, never on both at once.
_BOTH_SLIDES = [SlideRole.HE, SlideRole.IHC]

PIPELINE_STAGES: list[PipelineStage] = [
    PipelineStage(
        id="read-slide",
        index=1,
        title="Open the slide",
        tagline="Load the scan and pick the zoom level to work at.",
        what=(
            "A whole-slide image is not a photo, it is a pyramid: the same slide stored at "
            "many zoom levels, cut into tiles, so software can pull just the piece it needs."
        ),
        why_here=(
            "Everything downstream depends on one decision made here - what magnification do "
            "I work at? That choice is physical, expressed in microns per pixel (mpp), never "
            "as a level number, because different scanners have different native mpp."
        ),
        how="Library work: openslide-python for .svs/.ndpi/.mrxs, DeepZoom for browser tiles.",
        approach=Approach.LIBRARY,
        branch=Branch.SHARED,
        # Both, and genuinely at once: the demo reads the case's H&E and its IHC
        # slide together so the two readouts land side by side rather than the
        # second lagging a step behind. Hence `reads_slides_together` - there is no
        # "which slide" to choose here, both are already on screen.
        runs_on=list(_BOTH_SLIDES),
        reads_slides_together=True,
        input_label="The scanned slide file",
        output_label="Slide open, zoom level chosen",
        action_label="Open the slide",
        implemented=True,
        rule="Convert between levels using mpp, never using a hard-coded level index.",
        references=["OpenSlide, J Pathol Inform 2013"],
    ),
    PipelineStage(
        id="quality-control",
        index=2,
        title="Quality control",
        tagline="Find blurry, folded and pen-marked areas so they are not measured.",
        what=(
            "Slides have defects: out-of-focus areas, tissue folds, air bubbles, pen marks, "
            "dust and coverslip edges. Each one produces a confident, wrong answer downstream."
        ),
        why_here=(
            "QC runs before anything is measured or learned. An artefact that survives to the "
            "scoring stage is indistinguishable from biology by the time you see the number."
        ),
        how=(
            "Two pre-trained GrandQC models in sequence - tissue detection at 10 um/px, then "
            "multi-class artefact segmentation over the tissue it found - with classical "
            "HistoQC-style metrics (Tenengrad sharpness, RMS and Michelson contrast, Gabor "
            "texture) measured on the same patches to explain each call."
        ),
        approach=Approach.PRETRAINED,
        branch=Branch.SHARED,
        # An immunostained section folds, tears and picks up pen exactly as an H&E
        # does, and steps 11-16 measure on it. Quality-controlling only the slide
        # that is not scored was the largest single gap this field was added to close.
        runs_on=list(_BOTH_SLIDES),
        input_label="The open slide",
        output_label="A map of the damaged areas",
        action_label="Check slide quality",
        implemented=True,
        rule=(
            "GrandQC decides, the classical metrics explain. Nothing measured for the "
            "explanation is allowed back into the decision."
        ),
        references=["GrandQC, Nature Communications 2024", "HistoQC vs PathProfiler"],
    ),
    PipelineStage(
        id="tissue-mask",
        index=3,
        title="Find the tissue",
        tagline="Separate tissue from empty glass.",
        what="A binary mask that answers one question per pixel: is there tissue here at all?",
        why_here=(
            "Cheap and first, so every expensive step that follows runs on a smaller area. "
            "At this stage you only know tissue versus glass - fat is tissue and must be kept."
        ),
        how=(
            "Classical, no training: downsample to ~2 um/px, take the HSV saturation "
            "channel, subtract step 2's artefacts, then pick the cut from the remaining "
            "histogram - Otsu when it is two humps, Zack's triangle rule when it is one "
            "spike and a tail, which is what a weak counterstain produces. Then close, "
            "open and filter components, every distance in microns and every area in mm2, "
            "never in pixels."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.SHARED,
        # Already true before it was declared: step 4 asks for this slide's footprint
        # whichever slide it is handed, so steps 11 and 14 have always built an IHC
        # tissue mask. It just happened off-screen.
        runs_on=list(_BOTH_SLIDES),
        input_label="The slide, minus the damaged areas",
        output_label="A map of where the tissue is",
        action_label="Find the tissue",
        implemented=True,
        rule=(
            "Rule 1 - remove fat at the tissue-type step, not here. Fat is a semantic class, "
            "not a brightness threshold."
        ),
        references=[
            "Otsu, IEEE Trans SMC 1979",
            "Zack, Rogers & Latt, J Histochem Cytochem 1977",
            "BCSS, Grand Challenge",
        ],
    ),
    PipelineStage(
        id="white-calibration",
        index=4,
        title="Measure the blank glass",
        tagline="Learn what \"no stain\" looks like on this slide.",
        what=(
            "Sample the empty glass to estimate I0, the illumination intensity that corresponds "
            "to zero stain, per channel, for this slide."
        ),
        why_here=(
            "It has to happen before optical density, because OD is defined relative to I0. "
            "Every slide has its own I0 - lamp age, scanner and coverslip all shift it."
        ),
        how=(
            "Classical, no training: invert step 3's tissue mask, then discard three more "
            "things that are not glass - the outer frame, step 2's artefacts, and the ring "
            "just outside the section - and take the 95th percentile of each RGB channel "
            "over what is left. A quadratic illumination surface is fitted from patch "
            "samples alongside, and used instead of one flat value only when its swing "
            "across the slide exceeds its own residual scatter."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.SHARED,
        # The step where "per slide" is the whole argument rather than a detail: the
        # DAB optical density step 14 reports is divided by *this slide's* I0, so an
        # H&E white point would put the measurement on the wrong scale entirely.
        runs_on=list(_BOTH_SLIDES),
        input_label="The slide and its tissue map",
        output_label="The colour of blank glass on this slide",
        action_label="Measure the blank glass",
        implemented=True,
        rule=(
            "Calibration preserves your numbers. Normalisation overwrites them. Do not "
            "confuse them."
        ),
        references=[
            "Ruifrok & Johnston 2001",
            "Beer 1852",
        ],
    ),
    PipelineStage(
        id="optical-density",
        index=5,
        title="Measure stain amount",
        tagline="Turn colour into how much stain is present.",
        what=(
            "OD = -log10(I / I0). Optical density is linear in stain concentration; RGB "
            "intensity is not. Only in OD space can stains be un-mixed by linear algebra."
        ),
        why_here=(
            "This is the trunk both branches stand on. Concentration is linear here and "
            "nowhere else, so the measurement needs it - and so does the deconvolution that "
            "feeds the model, because that is a projection in this space. The fork itself is "
            "one step further on."
        ),
        how="Classical: Beer-Lambert transform, applied per channel.",
        approach=Approach.CLASSICAL,
        branch=Branch.SHARED,
        # Both arms need it and neither can borrow the other's: the model reads the
        # H&E's density, the measurement reads the IHC slide's. The staining verdict
        # this step produces is also per slide - it is how the pipeline knows which
        # of the two it is looking at.
        runs_on=list(_BOTH_SLIDES),
        input_label="A tile of the slide",
        output_label="A stain-amount picture of that tile",
        action_label="Measure stain amount",
        implemented=True,
        rule=(
            "Rule 2 - one deconvolution, two branches: the H channel to the model, the DAB "
            "channel to the measurement. Draw it as a Y, never as a line."
        ),
        references=["Ruifrok & Johnston 2001", "Beer 1852"],
    ),
    PipelineStage(
        id="colour-deconvolution",
        index=6,
        title="Colour deconvolution",
        tagline="Split the picture into the blue stain and the brown marker.",
        what=(
            "Every pixel is a mixture of blue nuclear counterstain and brown DAB. Deconvolution "
            "projects OD onto the two stain vectors and recovers one channel per stain. It runs "
            "once and serves both branches: the measurement thresholds DAB on an absolute OD "
            "scale, and the model is fed the haematoxylin channel in place of RGB."
        ),
        why_here=(
            "This is the fork, and it sits immediately after OD because both arms need it. A "
            "dark blue nucleus under faint brown looks, in RGB, like a moderately brown "
            "nucleus - you cannot threshold a mixture. And handing the model H instead of RGB "
            "is what lets one detector work on the H&E and on all five IHC slides: the brown, "
            "the single biggest source of colour nuisance, is gone before it can act."
        ),
        how=(
            "Classical: Ruifrok & Johnston, on fixed published vectors rather than per-image "
            "estimated ones, so the DAB scale means the same thing on every slide. The "
            "haematoxylin variation that survives is handled by HED augmentation when the "
            "model is trained, which costs nothing at inference; Macenko stays on the shelf as "
            "a fallback, applied to the H channel alone and only if validation asks for it."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.SHARED,
        # The step this field most obviously corrects. Shown on an H&E alone, its DAB
        # panel is a picture of nothing - there is no DAB in an H&E section - while
        # the IHC slide's DAB channel *is* what step 14 measures. The guide asks for
        # this one to be demonstrated on an IHC tile; until now it never was.
        runs_on=list(_BOTH_SLIDES),
        input_label="A stain-amount tile",
        output_label="A blue picture and a brown picture",
        action_label="Separate the stains",
        implemented=True,
        rule=(
            "Rule 3 - separate the stains before you threshold, and threshold the DAB "
            "channel only."
        ),
        references=["Ruifrok & Johnston 2001"],
    ),
    PipelineStage(
        id="tiling",
        index=7,
        title="Cut into tiles",
        tagline="Cut the tissue into small squares the model can read.",
        what=(
            "The slide is cut into 224 um squares - the window the trained model was "
            "fitted at - and every square is checked against the tissue mask and the "
            "artefact map before it is kept. The grid this step prices is the grid "
            "that actually runs, so the square count here IS step 8's cost."
        ),
        why_here=(
            "After QC and the tissue mask, so you only ever tile pixels worth "
            "processing - and before step 8, whose entire cost is the square count "
            "this screen reports."
        ),
        how=(
            "Plumbing: a tile index with coordinates, level, overlap and a "
            "tissue-fraction filter, laid at the checkpoint's own window read from its "
            "manifest. This screen used to ask which model and at what scale; the "
            "pipeline now commits to the colour head at 224 um, so it states the "
            "answer instead. The record is still written to disk so step 8, step 9 and "
            "the runner rebuild the same grid without being handed it."
        ),
        approach=Approach.PLUMBING,
        branch=Branch.MODEL,
        # H&E only, and stated rather than left to the default: this is where the
        # per-slide prefix ends and the region-finding arm begins.
        runs_on=[SlideRole.HE],
        input_label="The tissue area",
        output_label="A grid of tiles to check",
        action_label="Cut into tiles",
        implemented=True,
        rule=(
            "Tile after the tissue mask and after quality control, never before: every "
            "tile that survives is a forward pass through the region model. And the "
            "colour model this pipeline commits to reads a second dye, so it belongs on "
            "the H&E and nowhere else - on an immunostained section there is no eosin "
            "to read."
        ),
        references=["Slideflow, arXiv 2304.04142"],
    ),
    PipelineStage(
        id="tissue-type-segmentation",
        index=8,
        title="Tissue-type segmentation",
        tagline="Mark each tile as invasive tumour, tumour inside a duct, or not tumour.",
        what=(
            "Label every patch of tissue as one of three things: invasive tumour, tumour "
            "still inside a duct, or not tumour at all. This is the semantic map the whole "
            "score is gated on. A second pass over the finished map adds a fourth colour "
            "the model cannot produce itself - 'cannot be determined' - over the "
            "contained-tumour patches whose surroundings or shape do not support the "
            "call. It marks them for a human rather than changing the score, which was "
            "never measured on them. Before any answer is kept, patches that are blank "
            "scanner background, or unlike anything the model was trained on, are "
            "refused: they get no label, are drawn in the same colour, and leave every "
            "number."
        ),
        why_here=(
            "After tiling, before anything cellular. It is the expensive learned step, so it "
            "runs on the smallest area the cheap steps could give it."
        ),
        how=(
            "Inference only - this step trains nothing. Two frozen ResNet18 bodies, "
            "concatenated into a small MLP head, fitted offline and published with the "
            "input contract they were fitted under. Eight are published - four fields "
            "of view, 112 to 672 um, all at a 224 px window, times two input contracts "
            "- and the pipeline commits to ONE of them: the full-colour head at 224 um. "
            "The haematoxylin contract was the one meant to cover the H&E and all five "
            "IHC slides from a single model, and measured it does not - 0% invasive on "
            "every immunostained marker, at 0.96-0.98 confidence - so the generality it "
            "was paying for is generality step 10 supplies instead, by registration. "
            "On the H&E, where this now runs, the second dye is half the evidence."
        ),
        approach=Approach.TRAINED,
        branch=Branch.MODEL,
        trains_model=True,
        # H&E only, on evidence rather than on principle. The h_channel branch was
        # built to serve all six slides of a case, and measured on real IHC it does
        # not: 0% invasive on all five markers at 0.96-0.98 confidence, against 12.7%
        # on the same case's H&E. Silent, and therefore worse than a refusal. Step 10
        # exists because of this number - see `step12_ihc_alignment/README.md`.
        runs_on=[SlideRole.HE],
        input_label="The grid of tiles",
        output_label="A label for every tile",
        action_label="Label the tissue",
        implemented=True,
        rule=(
            "Rule 5 - DCIS and invasive tumour must be separate output classes. Only invasive "
            "tumour enters the denominator."
        ),
        references=[
            "UNI, Nature Medicine 2024",
            "Cruz-Roa, Sci Rep 2017",
            "Tellez, Medical Image Analysis 2019",
            "BCSS",
        ],
    ),
    PipelineStage(
        id="roi-mask",
        index=9,
        title="Build the ROI mask",
        tagline="Join the tumour tiles into clean regions.",
        what=(
            "Turn per-tile class probabilities into a single smooth, hole-free polygon that "
            "defines where scoring is allowed to happen. Alongside it, a second product read "
            "straight off step 8's own tile calls: adjacent same-class tiles merged into "
            "regions with a closed border drawn around each, for invasive, DCIS and the "
            "'cannot be determined' overlay - never for the non-epithelium bulk. The largest "
            "3 DCIS and 3 invasive regions are cropped from the slide itself, and every "
            "border-class region can be downloaded as a QuPath-importable GeoJSON file."
        ),
        why_here=(
            "Before nuclei segmentation, because the ROI is what makes nuclei segmentation "
            "affordable - seconds on a 5 % focus instead of GPU-hours on the whole slide. The "
            "per-class borders belong on the same screen because they are a second reading of "
            "the same class map step 8 just produced, not a new pass over the slide."
        ),
        how=(
            "Classical post-processing: smooth the invasive probability, threshold, close to "
            "merge nearby foci, carve confidently in-situ windows back out (Rule 5), drop "
            "components below a physical area. The per-class borders are a separate, "
            "parameter-free pass: connected-component labelling of step 8's raw class map, "
            "ranked by area, with the top regions cropped from the slide and exported as "
            "GeoJSON."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.SHARED,
        # H&E only: its input is step 8's class map, which exists for one slide.
        runs_on=[SlideRole.HE],
        input_label="The labelled tiles",
        output_label="Clean tumour outlines",
        implemented=True,
        action_label="Outline the tumour",
        rule="Rule 4 - segment the region before you segment the cells.",
        references=["Primer items 62-63, 70-72"],
    ),
    PipelineStage(
        id="roi-selection",
        index=10,
        title="Pick the regions to use",
        tagline="Choose which tumour regions go forward to the detailed check.",
        what=(
            "Step 9 traced every patch of invasive tile the model drew - typically 13 to 20 "
            "on a section. This screen lays them out as cards, each showing the tissue with "
            "the model's own square boundary on it, its area, and how confident the model "
            "was, and lets a person tick the ones that go forward. Nothing is segmented "
            "here and nothing is measured; the only thing this step produces is a choice."
        ),
        why_here=(
            "Immediately before the expensive step, because it is what makes the expensive "
            "step affordable. The next step runs a segmentation network per pixel, and "
            "running it over a whole section would be hours; running it over the regions "
            "ticked here is minutes. A coarse detector deciding where a precise one should "
            "look is the standard shape of this, and putting a person at the junction is "
            "what stops the pipeline from spending that budget on a fold or a pen mark."
        ),
        how=(
            "Plumbing over step 9's own connected components: rank them by area, give each "
            "a stable id, drop patches too small to hold a boundary, crop a card from the "
            "slide for each, and pre-tick enough of the largest to cover most of the "
            "tumour. The selection is stored against the class map it was made on, so a "
            "re-run of step 8 cannot leave a tick pointing at different tissue."
        ),
        approach=Approach.PLUMBING,
        branch=Branch.SHARED,
        # H&E only: the candidates are step 8's class map, which exists for one slide.
        runs_on=[SlideRole.HE],
        input_label="The tumour outlines",
        output_label="The regions you picked",
        implemented=True,
        action_label="Show the regions",
        rule=(
            "A coarse detector says where to look, never what the answer is. Nothing "
            "goes to the precise model that a person has not agreed to."
        ),
        references=["Primer items 62-63"],
    ),
    PipelineStage(
        id="roi-refinement",
        index=11,
        title="Refine the regions per pixel",
        tagline="Replace each rough square with the real tumour shape inside it.",
        what=(
            "Run BEETLE - a published segmentation network - on each chosen region on its "
            "own, and trace what it finds. Every region is shown as a before and after: the "
            "coarse square on the left, the boundary BEETLE drew inside it on the right, "
            "plus a flat map of what the network called every pixel of the box. Regions "
            "appear as they finish rather than at the end, one can fail without stopping "
            "the rest, and a failed one can be retried on its own."
        ),
        why_here=(
            "After a person has chosen and before anything crosses to the stained slide. A "
            "square drawn around tumour contains stroma, fat and glass, and every one of "
            "those is in the denominator if the square is what gets measured. This is the "
            "step where the tumour mask stops being a staircase of 224-micron windows and "
            "becomes the actual shape of the disease - and from here on it is the only "
            "tumour mask the pipeline uses."
        ),
        how=(
            "Pre-trained: BEETLE's released nnU-Net, at its own fixed 0.5 um/px, run over "
            "the windows of each padded candidate box and nowhere else. The per-pixel "
            "answer is reduced onto a 1 um/px canvas covering just that box, cleaned of "
            "specks and pinholes, and traced into polygons in the slide's own coordinates "
            "so the next step can warp them like any other region."
        ),
        approach=Approach.PRETRAINED,
        branch=Branch.SHARED,
        # H&E only: the boundary is traced where structure is legible, then carried
        # across by step 12 - the same argument step 8 makes, one model further on.
        runs_on=[SlideRole.HE],
        input_label="The regions you picked",
        output_label="Exact tumour outlines",
        implemented=True,
        action_label="Trace the edges",
        rule=(
            "Never run the precise model on the whole slide, and never let the coarse "
            "box be the answer once the precise one has spoken."
        ),
        references=[
            "BEETLE, Zenodo record 16812932",
            "Primer items 70-72",
        ],
    ),
    PipelineStage(
        id="ihc-alignment",
        index=12,
        title="Align the ROI to the IHC slide",
        tagline="Copy the tumour regions from the H&E slide onto the marker slide.",
        what=(
            "Register this case's H&E slide to the chosen marker's IHC slide and warp step "
            "11's pixel-level invasive boundary into the IHC slide's own coordinates. The "
            "two slides are shown side by side with those regions drawn as borders only - "
            "nothing filled - so the alignment can be judged before anything downstream "
            "depends on it, and each region is then cropped from the IHC slide itself."
        ),
        why_here=(
            "After the ROI exists and before anything is measured. Tissue typing is done on "
            "the H&E, where structure is legible and the model was trained; the brown that "
            "actually gets scored is on the IHC section. Those are two different physical "
            "slices of the same block, so the regions have to be carried across geometrically "
            "rather than assumed to share coordinates."
        ),
        how=(
            "VALIS: rigid registration then a non-rigid refinement, run in an isolated "
            "environment, with every region vertex warped through the resulting transform. A "
            "confidence gate (matched keypoints, residual error in microns, tissue-area "
            "ratio) refuses rather than emitting a mask it cannot stand behind, and the "
            "viewer confirms the overlay before the step completes."
        ),
        approach=Approach.LIBRARY,
        branch=Branch.SHARED,
        # Both, and the only step after step 1 that reads them *together* rather than
        # once each: registering a pair is the one operation that needs two slides in
        # hand at the same moment. Its own panel already shows them side by side.
        runs_on=list(_BOTH_SLIDES),
        reads_slides_together=True,
        input_label="The tumour outlines and the marker slide",
        output_label="The same regions, now on the marker slide",
        implemented=True,
        action_label="Match the slides",
        rule=(
            "Serial sections are not the same tissue. Refuse a registration you cannot "
            "measure rather than warping a mask onto the wrong cells."
        ),
        references=[
            "VALIS, Nature Communications 2023",
            "docs/segmentation-approaches-implementation.md Part 7",
        ],
    ),
    PipelineStage(
        id="nuclei-segmentation",
        index=13,
        title="Nuclei segmentation",
        tagline="Outline every cell nucleus inside the tumour regions.",
        what=(
            "Instance segmentation, not semantic: two touching nuclei must come out as "
            "two objects with their own boundaries, not one blob. It runs inside the "
            "three invasive regions step 10 put on the IHC slide, on a sample of fields "
            "spread across each one rather than on every pixel of them, and each nucleus "
            "comes back with its outline, its area and shape, and how much counterstain "
            "it holds."
        ),
        why_here=(
            "After the regions are on the IHC slide and before anything is measured. "
            "Nuclei models behave unpredictably on fat and necrosis, so running inside "
            "the region keeps junk detections out of the denominator - and the "
            "denominator is what this step exists to produce."
        ),
        how=(
            "Cellpose's published nuclei model (BSD-3-Clause), used as published with no "
            "training, on the CPU at 0.5 um/px, reading each field in grey. It was chosen "
            "on a benchmark of every alternative, including the earlier InstanSeg on the "
            "haematoxylin channel: it found the most of the cells that are really there, "
            "both against the matching H&E and against breast IHC cells labelled from "
            "immunofluorescence. A naive watershed runs on the same field beside it, "
            "because the difference between the two is what makes the case that instance "
            "segmentation is a separate problem."
        ),
        approach=Approach.PRETRAINED,
        branch=Branch.MEASUREMENT,
        # IHC only. It is addressed by the (H&E, IHC) pair - the regions came from the
        # H&E - but every pixel it segments is the immunostained slide's.
        runs_on=[SlideRole.IHC],
        input_label="The regions on the marker slide",
        output_label="One outline per cell",
        implemented=True,
        action_label="Find the cells",
        rule=(
            "Check the count against the H&E of the same block, per mm2 of tissue, on "
            "every pair. The detector sees the brown, so stained cells could in principle "
            "be found more readily than unstained ones; this check is what would show "
            "it. Squares that land on glass or scanner fill count as no tissue, not as "
            "missed cells."
        ),
        references=[
            "InstanSeg, arXiv 2024",
            "Cellpose, Nat Methods 2021",
            "StarDist, MICCAI 2018",
        ],
    ),
    PipelineStage(
        id="cell-typing",
        index=14,
        title="Sort the cells",
        tagline="Keep the tumour cells and set the other cells aside.",
        what=(
            "Not every cell inside a tumour region is a tumour cell. Lymphocytes, "
            "fibroblasts and endothelial cells are mixed in, and each one left in the "
            "denominator dilutes the percentage. This sorts step 11's nuclei by size, "
            "roundness, elongation and how dark the counterstain is, and reports how "
            "much of the answer each of those thresholds is deciding."
        ),
        why_here=(
            "Straight after segmentation and before anything is measured. This is the "
            "last filter on which cells count."
        ),
        how=(
            "Thresholds on nuclear morphology, every distance in microns - not a fitted "
            "model, because no nuclei in this project have been labelled and a model "
            "fitted on invented labels would carry the same judgements with a number "
            "attached that implied they were measured. The thresholds are live on the "
            "screen and the sensitivity of the mix to each one is published beside it."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.MEASUREMENT,
        runs_on=[SlideRole.IHC],
        input_label="The cell outlines",
        output_label="A type for every cell",
        implemented=True,
        action_label="Sort the cells",
        rule=(
            "A percentage needs a denominator somebody can defend. Say which cells are "
            "in it, and say how much the thresholds that decided are worth."
        ),
        references=[
            "HoVer-Net classification, MedIA 2019",
            "NuCLS, GigaScience 2022",
        ],
    ),
    PipelineStage(
        id="compartments",
        index=15,
        title="Build compartments",
        tagline="Mark the part of each cell where the marker should show up.",
        what=(
            "Nuclei segmentation gives you the nucleus. The stain is somewhere else. This "
            "grows each tumour cell outward from its nucleus into the region where the "
            "brown is supposed to be, and which region that is depends on the antibody: "
            "a ring around the membrane for CD44, ABCC4 and ABCC11, a band of cytoplasm "
            "for the two cadherins."
        ),
        why_here=(
            "After you know which cells count and before you measure anything. "
            "Compartments are geometry, not biology - but measuring the wrong part of "
            "the cell is not a small error, so the geometry has to be settled first."
        ),
        how=(
            "Classical morphology, every distance in microns and converted through the "
            "slide's own scale. Each cell grows only as far as the midline between it "
            "and its neighbours - a Voronoi constraint - so no two cells can claim the "
            "same pixel and a strongly stained cell cannot bleed its signal into a "
            "negative neighbour. Neither width is established fact, so the sensitivity "
            "of the compartment to the width is published beside it."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.MEASUREMENT,
        runs_on=[SlideRole.IHC],
        input_label="The tumour cells and the marker name",
        output_label="A measuring area for every cell",
        implemented=True,
        action_label="Mark the areas",
        rule=(
            "This is the first step that reads the antibody letter. A thin membrane ring "
            "on a cytoplasmic marker reports how tightly the tissue is packed, not how "
            "much stain the cell made."
        ),
        references=[
            "QuPath, Scientific Reports 2017",
            "primer items 44-45, 88-89",
        ],
    ),
    PipelineStage(
        id="per-cell-measurement",
        index=16,
        title="Measure each cell",
        tagline="Measure how much brown marker each cell has.",
        what=(
            "One number always: the mean DAB optical density in that cell's compartment. Plus, "
            "for the three membrane markers only, a second number - how complete the ring is, "
            "over 36 bins of 10 degrees. The two cytoplasmic markers get a different second "
            "number: what share of the band's pixels are brown."
        ),
        why_here=(
            "This is the first moment anything is actually measured; everything before it was "
            "deciding what to measure. And it runs on raw calibrated pixels, straight off step "
            "6's DAB channel - this is the step Rule 2 exists to protect. Nothing between step "
            "4 and here rewrites them."
        ),
        how=(
            "Classical arithmetic: mask the DAB channel by step 13's compartment and take the "
            "mean - mean rather than maximum, for every marker, so two markers' intensities can "
            "be compared. Then walk around the ring in 36 angular bins and count the stained "
            "ones, or, for a cytoplasmic marker, take the stained share of the band's pixels."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.MEASUREMENT,
        # IHC only, and the step that makes the whole prefix matter: the DAB channel
        # it reads is this slide's own unmix, divided by this slide's own white point.
        runs_on=[SlideRole.IHC],
        input_label="The measuring areas and the brown picture",
        output_label="One measurement per cell",
        action_label="Measure the cells",
        implemented=True,
        rule=(
            "Completeness is not computed for a cytoplasmic marker at all. Cytoplasmic staining "
            "has no circumference, so a shared code path quietly feeding a near-zero value into "
            "the positivity rule would make a cell negative for a geometry it never had."
        ),
        references=[
            "ASCO/CAP HER2 completeness criteria",
            "QuPath IHC cell classification",
            "IHC Profiler, PMC",
            "DeepLIIF, Nature Machine Intelligence 2022",
        ],
    ),
    PipelineStage(
        id="intensity-binning",
        index=17,
        title="Intensity binning",
        tagline="Give each cell a strength grade: 0, 1+, 2+ or 3+.",
        what=(
            "Two things happen here, and conflating them is a common bug. Each cell gets 0 / 1+ "
            "/ 2+ / 3+, which is internal machinery for counting positives and for the H-score. "
            "The case's reported intensity is a different scale - OncoStem's 0 to 2, with "
            "permitted values 0, 0.5, 1, 1.5, 1.75 and 2 - and it is decided once, at step 16."
        ),
        why_here=(
            "After per-cell measurement, before aggregation. This is the translation layer "
            "between the machine's world and the pathologist's."
        ),
        how=(
            "Classical: absolute cut points on the calibrated optical-density scale, five "
            "independent sets in one versioned config file. Each antibody has its own "
            "concentration, incubation time and detection chemistry, so the density that means "
            "'moderate' for CD44 is not the one that means 'moderate' for pan-cadherin - and "
            "the observed data says so: CD44 spans 5 to 85 per cent across six cases while "
            "N-cadherin is 80 on every single read."
        ),
        approach=Approach.CLASSICAL,
        branch=Branch.MEASUREMENT,
        runs_on=[SlideRole.IHC],
        input_label="The per-cell measurements",
        output_label="A grade for every cell",
        action_label="Grade the cells",
        implemented=True,
        rule=(
            "Per-slide percentiles normalise away the diagnosis: a weak slide and a strong one "
            "score identically. Use absolute, calibrated cuts, anchored on controls if you can "
            "get them."
        ),
        references=[
            "Primer items 49, 79-82",
            "Ruifrok & Johnston 2001",
            "images-to-scores-mapping.md, Parts 4 and 6",
        ],
    ),
    PipelineStage(
        id="aggregate",
        index=18,
        title="Aggregate into a score",
        tagline="Combine the cells into the two numbers reported for this marker.",
        what=(
            "Percent positive - 100 x positive tumour cells over total tumour cells, rounded to "
            "the nearest 5 - and intensity, the mean DAB optical density over the positive cells "
            "mapped onto the nearest permitted band. That is the whole deliverable. The H-score, "
            "the Allred score and the ASCO/CAP category are computed and shown beside it as the "
            "field's standard vocabulary, never as the output."
        ),
        why_here=(
            "Last. By now every hard decision has been made, and no scoring logic reaches back "
            "into pixels."
        ),
        how=(
            "Pure logic over step 14's stored rows: arithmetic and if statements, in a small, "
            "separately reviewable module that imports nothing from the service layer - so a "
            "disputed number can be re-derived from the JSON with a calculator."
        ),
        approach=Approach.LOGIC,
        branch=Branch.MEASUREMENT,
        runs_on=[SlideRole.IHC],
        input_label="The graded cells",
        output_label="Percent positive and staining strength",
        action_label="Calculate the score",
        implemented=True,
        rule=(
            "Both roundings are load-bearing, not cosmetic. Real reported percentages are "
            "multiples of 5, and 119 of 120 real intensities land exactly on a band value - so "
            "61.7 per cent and 1.34 read as a different measurement from the one that was asked "
            "for, not as a more precise one."
        ),
        references=[
            "images-to-scores-mapping.md, Parts 4 and 6",
            "ASCO/CAP HER2 guideline, summarised",
            "QuPath TMA scoring validation, PubMed",
            "CD44/CD24 cutoffs, PLOS One",
        ],
    ),
    PipelineStage(
        id="validation",
        index=19,
        title="Compare with pathologists",
        tagline="Check how close the score is to what pathologists reported.",
        what=(
            "Compare model scores against pathologist scores on a held-out set, and against "
            "the agreement pathologists reach with each other."
        ),
        why_here=(
            "After the score exists. Validation is what converts an output into a claim, and "
            "it is the only step that can tell you the pipeline order was right."
        ),
        how="Statistics: Bland-Altman, weighted kappa, correlation stratified by tumour content.",
        approach=Approach.LOGIC,
        branch=Branch.SHARED,
        # Neither slide: agreement is five markers of one block against four readers
        # of the same block, so a single slide is not a thing this can be run on.
        runs_on=[SlideRole.CASE],
        input_label="Our scores and the pathologists' scores",
        output_label="How closely they agree",
        action_label="Compare the scores",
        implemented=True,
        rule=(
            "Compare against inter-pathologist agreement, not against a perfect ground truth. "
            "No reading in the 120 real percent values sits more than 10 points from its case "
            "mean, so plus or minus 10 absolute points is the human spread - and the standard."
        ),
        references=["QuPath validation vs manual scoring", "Cruz-Roa, Sci Rep 2017"],
    ),
]

STAGES_BY_ID: dict[str, PipelineStage] = {stage.id: stage for stage in PIPELINE_STAGES}
