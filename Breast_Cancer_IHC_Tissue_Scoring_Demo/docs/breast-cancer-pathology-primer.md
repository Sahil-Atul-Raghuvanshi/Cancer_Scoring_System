# Breast Cancer & Digital Pathology — A Primer for Data Scientists

Written for someone with a data science background and no biology beyond school-level basics.
Goal: go from zero to understanding exactly what this codebase's segmentation and scoring pipeline does, and why.

100 items, sequenced from absolute basics to advanced. Each item is short and self-contained — read straight through, or jump to a stage.

---

## Stage A — Biology Foundations

### 1. What is a cell
The smallest living unit of the body — the basic "object instance" everything biological is built from. A human body has roughly 37 trillion of them. Each cell takes in nutrients, produces energy, and can divide to make more cells.

### 2. Cell structure: nucleus, cytoplasm, membrane
- **Membrane** — the outer wall of the cell; controls what goes in/out. (CD44, the marker in your screenshots, sits on this wall — that's why it's called a "membrane marker.")
- **Cytoplasm** — the fluid-filled interior where most cell machinery operates.
- **Nucleus** — a compartment holding the DNA, the cell's instruction manual. Nuclei stain darkly on slides, which is why pathologists (and the code) use nucleus visibility to find/count cells.

### 3. What is a tissue
A group of similar cells organized to do one job — e.g. skin tissue, muscle tissue. Like a group of same-typed objects forming a subsystem.

### 4. Types of tissue (epithelial, connective, muscle, nervous)
- **Epithelial** — lining/covering tissue (skin surface, gland ducts). Breast cancer starts here.
- **Connective** — structural/supportive tissue (fat, fibrous tissue, blood). This is what "stroma" means later on.
- **Muscle** — contracts, generates movement.
- **Nervous** — transmits signals.
You mainly need *epithelial* (where cancer starts) and *connective/stroma* (the packing material around it).

### 5. What is an organ and organ system
An organ (e.g. the breast) is multiple tissue types working together for one function. An organ system is multiple organs working together (e.g. the reproductive system).

### 6. What is DNA, genes, and proteins (super basic)
- **DNA** = the instruction manual, stored in the nucleus.
- A **gene** = one specific recipe within that manual (e.g. "recipe for CD44 protein").
- A **protein** = the functional molecule built from that recipe — proteins *do things* (structure, signaling, receptors). CD44, ER, HER2 are all proteins. IHC staining detects proteins, not DNA directly.

### 7. Cell division: mitosis basics
Normal cells divide in a controlled way to replace old/damaged cells, like a managed process with checks before it's allowed to run. **Ki67** is a protein present only in actively-dividing cells, so staining for it measures what fraction of cells are currently dividing — a direct proxy for how aggressively a tumor is growing.

### 8. What is a mutation
A copying error in the DNA instructions — a bug introduced during cell division. Most mutations are harmless or get repaired. Some disable the checks that normally control cell division.

### 9. What is cancer (uncontrolled cell growth)
Cells whose division-control mutations have accumulated enough that they divide continuously, ignoring normal stop signals — a runaway process with no governor. They also gain the ability to invade neighboring tissue and spread (metastasize).

### 10. Benign vs malignant tumors
- **Benign** — abnormal growth that stays contained, doesn't invade or spread. Not dangerous cancer.
- **Malignant** — invades surrounding tissue and can spread elsewhere (metastasis). This is what "cancer" means clinically. This *invasion* distinction is exactly why the app has an "invasive tumour region" step — it isolates the part of the tissue where cells have broken through and become malignant, as opposed to normal or pre-invasive tissue.

---

## Stage B — Breast Anatomy & Breast Cancer Basics

### 11. Basic anatomy of the breast (lobules, ducts, stroma)
The breast is mostly: **lobules** (small clusters that produce milk), **ducts** (tubes that carry milk to the nipple), and **stroma** (the fatty/fibrous connective tissue filling the rest of the space). Nearly all breast cancers arise from the cells lining the ducts or lobules.

### 12. What are breast lobules and ducts
Lobules = grape-cluster-shaped glands that make milk. Ducts = the tube network connecting lobules to the nipple, lined by a single layer of epithelial cells. Cancer starts in this epithelial lining.

### 13. What is breast cancer
Uncontrolled growth of the epithelial cells lining the ducts or lobules, forming a mass (tumor) that can stay contained or invade surrounding tissue.

### 14. Where breast cancer originates (ductal vs lobular)
- **Ductal** — arises in the duct lining. ~80% of breast cancers (this is what "IDC," Invasive Ductal Carcinoma, refers to).
- **Lobular** — arises in the lobule lining (~10%, "ILC").
Naming reflects where the cancer started, not necessarily where it currently is.

### 15. Ductal Carcinoma In Situ (DCIS) vs Invasive Carcinoma
- **In situ** ("in place") — abnormal cells are confined inside the duct, haven't broken through its wall. Not yet invasive; much lower risk.
- **Invasive** — cells have broken through the duct wall into surrounding stroma, gaining access to blood/lymph vessels and the ability to spread.
This is the single most important distinction for your pipeline: scoring must be measured on the *invasive* region only, excluding DCIS, because DCIS biology and treatment implications differ.

### 16. Stages of breast cancer (TNM staging basics)
**T**umor size, **N**ode involvement (has it reached lymph nodes), **M**etastasis (has it spread to distant organs). Combined into a stage number (0–IV) that summarizes how advanced the disease is. Distinct from *grade* (see next item).

### 17. Grades of breast cancer (differentiation)
How abnormal the cancer cells look and behave compared to normal cells — "differentiation." Grade 1 (well-differentiated, look closer to normal, usually slower-growing) through Grade 3 (poorly differentiated, look very abnormal, usually faster-growing). Grade is about cell appearance/behavior; stage is about extent of spread.

### 18. Types of breast cancer (IDC, ILC, triple-negative, HER2+)
Beyond ductal/lobular origin, cancers are subtyped by which biomarkers they express:
- **HER2+** — overexpresses HER2 protein, targetable by HER2-blocking drugs.
- **Triple-negative** — lacks ER, PR, and HER2 expression; harder to target, historically worse prognosis.
- **Hormone-receptor-positive (ER+/PR+)** — responds to hormone-blocking therapy.
This is *why* biomarker scoring exists at all: the subtype determines treatment.

### 19. Risk factors and epidemiology basics
Age, family history/genetic mutations (e.g. BRCA1/2), hormonal exposure, and lifestyle factors all influence risk. Not needed for the pipeline itself, but useful context for why this is such a heavily studied disease.

### 20. Why molecular subtyping matters for treatment
Two tumors that look identical under a microscope can behave completely differently depending on their molecular markers. A HER2+ tumor responds to trastuzumab; a triple-negative one doesn't. This is the entire reason IHC marker testing (ER/PR/HER2/Ki67/CD44...) exists — it's what turns "cancer" into "*this specific* cancer, treat it *this specific* way."

---

## Stage C — Diagnosis Workflow

### 21. How breast cancer is diagnosed (biopsy, imaging)
Typically: imaging (mammogram/ultrasound/MRI) finds a suspicious area → a biopsy removes a small tissue sample → that sample is examined under a microscope to confirm cancer and characterize it.

### 22. What is a biopsy and how tissue is obtained
A small sample of tissue extracted with a needle (core biopsy) or surgically (excisional biopsy), sent to a pathology lab for processing and examination.

### 23. From biopsy to slide: tissue processing (fixation, embedding, sectioning)
The raw tissue sample goes through: **fixation** (preserved in formalin so it doesn't decay), **embedding** (set in a wax block for stability), **sectioning** (sliced into micrometer-thin slices with a microtome), then mounted on a glass slide. This is why pathology slides are called "FFPE" (formalin-fixed, paraffin-embedded) samples.

### 24. What is a pathology lab workflow
Sample intake → processing (see above) → staining → pathologist review under microscope (or, now, digital scan + review) → report generation → results sent to the treating oncologist.

### 25. Role of the pathologist in diagnosis
A physician specialized in reading tissue slides — they confirm whether cancer is present, classify its type/grade, assess biomarkers, and estimate the extent of invasion. Their report directly drives treatment decisions.

### 26. What is a pathology report
A structured document summarizing the pathologist's findings: tumor type, grade, size, margins, biomarker scores (ER/PR/HER2/Ki67 etc.), and any other relevant observations. This is the clinical "output" this whole system is ultimately trying to help produce or support.

### 27. Multidisciplinary tumor board basics
A team of oncologists, surgeons, radiologists, and pathologists who review a patient's case together and agree on a treatment plan, informed heavily by the pathology report's biomarker findings.

### 28. Why biomarker testing matters for treatment decisions
Because treatment is now targeted, not one-size-fits-all: hormone therapy for ER+/PR+, HER2-blocking drugs for HER2+, chemotherapy considerations informed by Ki67/proliferation markers, and so on. An inaccurate biomarker score can lead to a patient getting (or missing) the wrong therapy — which is exactly why scoring accuracy and validation (Stage L) matter so much for a tool like this one.

---

## Stage D — Histology & Staining

### 29. What is histology
The study of tissue structure under a microscope. "Histopathology" = histology applied to diagnosing disease.

### 30. What is a microscope slide
A thin glass plate holding a tissue section, viewed under a microscope (traditionally) or scanned into a digital image (in this app's case).

### 31. H&E staining basics (Hematoxylin and Eosin)
The default, near-universal stain for tissue slides. Unstained tissue is nearly transparent/colorless under a microscope — H&E adds contrast so structures become visible.

### 32. What hematoxylin stains (nuclei, blue/purple)
Hematoxylin binds to DNA/RNA-rich structures — mainly the **nucleus** — staining it blue-purple. This is why nuclei appear dark and countable in every stained image you'll see, including in this pipeline's segmentation logic.

### 33. What eosin stains (cytoplasm/stroma, pink)
Eosin stains proteins in the cytoplasm and connective tissue pink/red, giving structural contrast around the nuclei.

### 34. What is immunohistochemistry (IHC)
A staining technique that detects a *specific protein* (not just generic structures) using an antibody that binds only to that protein, then produces a visible colored signal at that location. This is how the app measures "how much CD44 is present, and where."

### 35. How IHC antibodies work (antigen-antibody binding)
An **antibody** is a molecule engineered/selected to bind specifically to one target protein (the "antigen") — like a lock that only fits one key. In IHC, the antibody is chosen to bind the biomarker of interest (e.g. CD44), then a chemical reaction attached to the antibody produces color only where it has bound.

### 36. What is DAB staining (brown chromogen)
DAB (3,3'-Diaminobenzidine) is the chemical that turns brown wherever the marker-specific antibody has bound. Brown = "the marker is present here." The darker/more brown, the more marker present — this intensity is what the scoring math measures.

### 37. Why IHC uses a counterstain (hematoxylin + DAB together)
DAB alone only shows *where the marker is*, not *where the cells are* — you need to see the tissue structure to interpret it. So IHC slides are still counterstained with hematoxylin (nuclei = blue/purple) alongside DAB (marker = brown), giving both "where are the cells" and "where is the marker" in one image. This dual-stain is exactly what the color-deconvolution step in the code is untangling.

### 38. Common IHC biomarkers overview (ER, PR, HER2, Ki67, CD44)
A short reference table you'll want to remember:

| Marker | Location in cell | What it tells you |
|---|---|---|
| ER | Nucleus | Hormone-receptor status, guides hormone therapy |
| PR | Nucleus | Same, secondary hormone receptor |
| HER2 | Membrane | Guides HER2-targeted therapy |
| Ki67 | Nucleus | Proliferation rate (how fast the tumor grows) |
| CD44 | Membrane | Associated with invasion/stem-like tumor cells |

---

## Stage E — Specific Biomarkers

### 39. What is ER (Estrogen Receptor) and why it matters
A protein inside the nucleus that responds to estrogen. If present (ER+), the tumor's growth may be fueled by estrogen — and can be treated by blocking estrogen (hormone therapy).

### 40. What is PR (Progesterone Receptor)
Similar to ER but for progesterone. Usually tested alongside ER; PR+ status often correlates with a better response to hormone therapy.

### 41. What is HER2 and HER2-targeted therapy
HER2 is a membrane receptor protein that, when overexpressed, drives aggressive cell growth. HER2+ tumors are treated with drugs like trastuzumab (Herceptin) that specifically block it — a landmark example of biomarker-guided ("precision") cancer treatment.

### 42. What is Ki67 (proliferation marker)
A nuclear protein present only in cells actively going through cell division. The percentage of Ki67-positive cells estimates how fast the tumor is proliferating — high Ki67 generally means faster-growing, more aggressive disease.

### 43. What is CD44 and its role in breast cancer (stemness/invasion marker)
CD44 is a membrane protein involved in cell adhesion and migration. It's associated with "cancer stem cell"-like behavior and with the tumor's ability to invade surrounding tissue — which is why this platform pairs CD44 scoring specifically with the "invasive tumour region," since that's the biologically relevant area for this marker.

### 44. Membrane markers vs nuclear markers vs cytoplasmic markers
Where a marker's protein physically sits inside/on the cell:
- **Membrane** (CD44, HER2) — on the cell's outer wall, appears as a ring/border around the cell.
- **Nuclear** (ER, PR, Ki67) — inside the nucleus, appears as a filled blob.
- **Cytoplasmic** — spread through the cell interior.
This matters enormously for scoring math, because the shape you're looking for in the image is different for each (a ring vs. a blob vs. diffuse fill).

### 45. Why marker location changes how you score it
A nuclear marker just needs "is the nucleus brown or not, how dark." A membrane marker needs the code to check for a *ring-shaped* stain pattern that traces the cell boundary — a much harder geometric detection problem. This is exactly why `membrane.py` in this codebase has special "ring completeness" logic that a nuclear scorer wouldn't need.

### 46. Companion diagnostics concept (biomarker test tied to a drug)
A "companion diagnostic" is a test explicitly designed to determine whether a patient is eligible for a specific drug (e.g., a HER2 test determines HER2-drug eligibility). This regulatory category is why biomarker tests are held to strict accuracy/validation standards — a wrong result can directly deny or misassign a therapy.

---

## Stage F — Pathologist Scoring Standards

### 47. What is IHC scoring
The process of converting a stained slide's visual pattern into a number or category that summarizes "how much marker, how strong" — e.g., percent of cells positive, and how intensely stained.

### 48. What is percent positivity
The fraction of relevant cells that show marker staining above a positivity cutoff — e.g., "37% of tumor cells are CD44-positive."

### 49. What is staining intensity grading (0/1+/2+/3+)
A visual scale pathologists use to grade *how dark* the stain is per cell or per field: 0 = none, 1+ = weak, 2+ = moderate, 3+ = strong. This project uses an equivalent grading (Negative/Weak/Moderate/Strong).

### 50. What is H-score and how it's calculated
A composite score: `H-score = (% cells at 1+ × 1) + (% cells at 2+ × 2) + (% cells at 3+ × 3)`, ranging 0–300. Combines both how many cells are positive *and* how strongly, into a single number.

### 51. What is Allred score and how it's calculated
Another composite: a **proportion score** (0–5, based on % positive cells) plus an **intensity score** (0–3), summed to 0–8. Historically common for ER/PR scoring.

### 52. ASCO/CAP guidelines for biomarker scoring
The American Society of Clinical Oncology and College of American Pathologists jointly publish standardized scoring guidelines (e.g., what counts as HER2-positive) so labs worldwide score consistently. These guidelines are the "spec" pathology labs are validated against.

### 53. Inter-observer variability in manual scoring
Two pathologists looking at the same slide can produce meaningfully different scores — staining is subjective at the margins. This variability is a known, accepted limitation of manual IHC scoring, and it's the benchmark any automated tool has to be compared against (you can't expect a computer to beat a standard that itself has built-in disagreement).

### 54. Why pathologists use consensus scoring (multiple readers)
To reduce the effect of individual variability, important cases are scored by multiple pathologists and averaged/reconciled — exactly the "minimum 3 pathologists, reconcile if >10% off the mean" rule documented for this project (EPIC-009).

### 55. What is a reference standard / ground truth in pathology
The "correct answer" a new test is measured against — usually consensus pathologist scoring, since there's rarely a more objective alternative. All validation in Stage L is about comparing the algorithm's output to this reference standard.

### 56. Concordance and discordance in scoring
**Concordance** = the automated/alternative score agrees with the reference standard (within an accepted tolerance). **Discordance** = it doesn't. Validation studies report concordance rates as their headline result.

---

## Stage G — Digital Pathology Basics

### 57. What is digital pathology
The practice of scanning physical glass slides into high-resolution digital images, and reading/analyzing those images instead of (or alongside) looking through a physical microscope.

### 58. What is whole slide imaging (WSI) and slide scanners
A **slide scanner** is a specialized machine that photographs an entire glass slide at high magnification, stitching it into one enormous image — a "whole slide image" (WSI). This is the raw input your app receives as an uploaded slide.

### 59. What is a whole slide image file format (pyramidal, tiled)
Because WSI files are huge, they're stored as an image "pyramid" — the same image saved at multiple zoom levels, each level split into small tiles. This lets viewers load only the tiles needed for the current zoom/pan, instead of the entire gigapixel file.

### 60. Why whole slide images are huge (gigapixel images)
A single slide scanned at diagnostic resolution can be tens of thousands of pixels per side — often 1–10+ gigapixels total, multiple gigabytes per file. This is why the pipeline processes images in tiles rather than loading the whole thing into memory at once (see "tiled per-cell scoring" in EPIC-010).

### 61. What is a tissue mask
A binary map marking "tissue here" vs. "empty glass/background here," used to restrict all downstream processing to the actual specimen and ignore blank slide area — this is the `tissue_mask` function referenced in the codebase.

### 62. What is region-of-interest (ROI) selection
Narrowing analysis to a specific sub-area of the slide relevant to the question at hand (e.g., just the invasive tumor, not the whole tissue). Equivalent to filtering a dataset down to the relevant rows before running an analysis.

### 63. Why invasive tumour region matters vs whole tissue
Scoring over the *entire* tissue would mix in DCIS, stroma, immune cells, and normal tissue — diluting or corrupting the biomarker signal that's only clinically meaningful within invasive cancer cells. Gating to the invasive region first is a data-cleaning step as much as a biological one.

### 64. What is computational pathology (CV + AI applied to slides)
The broader field of applying computer vision and machine learning to digital pathology images — spanning classical image processing (what this app currently uses) through deep learning models (what it could use later).

---

## Stage H — Image Processing Fundamentals

### 65. What is an image in terms of pixels and color channels
An image is a grid of pixels, each holding numeric color values — typically 3 numbers per pixel (Red, Green, Blue). A stained slide image is just a big matrix of these RGB triples, which is what all the "math" in this pipeline actually operates on.

### 66. RGB color space basics
The standard way to represent color as three intensity values (Red, Green, Blue), each 0–255. Straightforward, but not ideal for isolating specific stains, which is why the pipeline converts into other representations (HSV, then HED — see Stage I).

### 67. HSV color space basics
An alternative color representation: **H**ue (which color), **S**aturation (how vivid/pure vs. washed out), **V**alue (how bright/dark). Useful for tissue detection because plain glass/background tends to have very low saturation compared to stained tissue — this is exactly what `tissue_mask` uses to separate tissue from background.

### 68. What is thresholding (binary image creation)
Converting a grayscale/intensity image into black-and-white by picking a cutoff value: anything above it is 1 ("yes"), anything below is 0 ("no"). Nearly every decision in this pipeline (is this pixel a nucleus? is this pixel stained?) ultimately reduces to a threshold.

### 69. What is Otsu's method for automatic thresholding
An algorithm that automatically picks the "best" threshold for separating an image into two classes (e.g., foreground vs. background) by finding the cutoff that best separates the image's intensity histogram into two peaks — no manual tuning needed. Used as a fallback threshold method in this codebase.

### 70. What are morphological operations (erosion, dilation, opening, closing)
Operations that reshape a binary mask:
- **Erosion** — shrinks regions (strips away thin/edge pixels).
- **Dilation** — grows regions.
- **Opening** (erode then dilate) — removes small isolated specks/noise.
- **Closing** (dilate then erode) — fills small holes.
The segmentation code uses opening/closing to clean up the tumor mask into a coherent shape, and erosion to detect thin membrane rings.

### 71. What are connected components / contours
Once you have a binary mask, "connected components" groups touching foreground pixels into distinct blobs/objects, and "contours" trace the outline around each blob. This is how a mask of "positive pixels" becomes "N distinct cells" or how a tumor mask becomes the green outline you see in the UI.

### 72. What is Gaussian smoothing/blurring
Averaging each pixel with its neighbors, weighted by distance, to reduce noise and small-scale variation — like a moving average in a time series, but in 2D. The segmentation code blurs the nucleus mask before thresholding, turning individual nucleus dots into a smooth "density" surface.

### 73. What is a density map (kernel density estimation)
A map showing *how concentrated* something is per local area, rather than just where individual points are — conceptually the same idea as KDE in statistics, applied to an image. The pipeline's "nuclear density map" is exactly this: how packed with nuclei is each neighborhood, used as a proxy for "is this tumor tissue."

### 74. What is optical density in imaging
A measure of how much light is absorbed passing through a sample — physically grounded (based on the Beer-Lambert law, Stage I), rather than just "how dark does this pixel look" in raw RGB terms. Optical density is what actually correlates with stain *concentration*, which is why more rigorous stain-quantification math (EPIC-010's `stains.py`) converts to optical density before thresholding.

---

## Stage I — Stain-Specific Color Math

### 75. What is color deconvolution
The mathematical process of "un-mixing" a multi-stained image back into separate channels, one per stain — e.g., splitting a combined hematoxylin+DAB image into a pure "hematoxylin amount per pixel" channel and a pure "DAB amount per pixel" channel. This is the `rgb2hed` step used throughout the codebase.

### 76. What is the Beer-Lambert law and its role in stain quantification
A physical law stating that light absorbance is proportional to the concentration of the absorbing substance and the distance light travels through it. Applied to stained tissue, this means the "darkness" you see is mathematically related to *how much stain* is there — giving color deconvolution its physical justification rather than being an arbitrary color trick.

### 77. Ruifrok & Johnston H-DAB deconvolution method
A well-established, published method (Ruifrok & Johnston, 2001) that defines standard reference color vectors for hematoxylin and DAB, enabling reliable separation of the two stains from an RGB image via the Beer-Lambert relationship. This is the named method referenced in the codebase's EPIC-010 stain work.

### 78. Why stain reference vectors matter
A "reference vector" defines the expected exact color signature of a pure stain (e.g., "pure hematoxylin looks like this RGB direction"). Using a fixed, standard reference vector (rather than guessing from each image) makes stain separation consistent and comparable across different slides/scanners — critical for the "absolute stain scale" work in EPIC-010.

### 79. Per-slide normalization vs absolute stain scale
- **Per-slide normalization** — rescale each slide's stain intensities based on that slide's own min/max (e.g. 1st–99th percentile). Simple, but means "37% intensity" on one slide isn't necessarily comparable to "37%" on another.
- **Absolute stain scale** — calibrate against a fixed physical reference (e.g. a blank glass area) so intensity values are comparable *across* slides and scanners.
EPIC-010 is specifically about moving from the former to the latter, for more clinically trustworthy, comparable numbers.

### 80. Why white-balance/reference calibration matters
Different scanners, lighting, and staining batches can shift the baseline color of an image even before any stain is applied. Calibrating against a known white/blank reference area corrects for this drift — the same reason cameras white-balance before capturing a photo.

### 81. What is a stain intensity histogram
A plot of how many pixels fall at each intensity level for a given stain channel. Useful for visually/statistically choosing thresholds — e.g., seeing two humps in the DAB histogram (stained vs. unstained population) helps validate that a chosen cutoff sits in the right place.

### 82. How percent positive is computed from stained pixel counts
At its simplest: count pixels/cells whose stain-channel value exceeds the positivity threshold, divide by the total relevant pixels/cells, multiply by 100. Everything upstream (deconvolution, normalization, thresholding) exists purely to make this final count trustworthy.

---

## Stage J — Classical Segmentation Approach (this project, today)

### 83. What is a heuristic algorithm (rule-based, non-ML)
An algorithm built from explicit, human-designed rules and fixed thresholds (e.g., "if nuclear density > 0.12, call it tumor") rather than rules learned automatically from data. Deterministic, explainable, and auditable — but only as good as the assumptions the human encoded.

### 84. How nuclear density is used as a tumour-region proxy
The logic: tumor tissue tends to be densely packed with abnormal nuclei, more so than normal stroma or sparse tissue. So "how many nucleus-pixels per local area" becomes a stand-in for "is this tumor" — an approximation, not a direct measurement of malignancy.

### 85. Why adipose/fat tissue needs to be excluded
Fat cells are large and mostly empty (they store fat, not much stainable protein), so they appear as large pale/white regions with very few nuclei. Left unfiltered, these areas could be mistaken for "background" or skew density calculations — so the heuristic explicitly excludes near-white regions before finalizing the tumor mask.

### 86. What "confidence = 0" means for a heuristic detector
The code hard-codes the segmentation's reported confidence to 0.0, as an explicit signal that this number isn't a statistical probability from a trained model — it's a rule-based estimate with no learned uncertainty quantification behind it. It's a deliberately honest "don't over-trust this number" flag.

### 87. Limitations of heuristic segmentation (DCIS vs invasive confusion)
Nuclear density alone can't distinguish DCIS (dense nuclei confined inside a duct) from truly invasive cancer (dense nuclei that have broken into stroma) — both look "densely packed" to this algorithm. This is the main known weakness flagged for validation, and the main reason a trained model might eventually be needed for this specific step.

### 88. What membrane completeness/ring detection means
For a membrane marker like CD44, a "real" positive cell should show a thin, continuous stained ring tracing its outer boundary. The code checks how much of that expected ring is actually stained (a "completeness" percentage) to distinguish genuine membrane staining from noise or partial/non-specific staining.

### 89. How positive-cell counting works geometrically
Rather than just counting individual stained pixels, the scorer identifies cell-sized/shaped structures (via the ring/completeness check), and counts whole cells as positive or negative based on whether their ring meets the completeness bar — closer to how a pathologist visually judges "is this cell's membrane properly stained," not just "is there brown pigment somewhere nearby."

### 90. Why gating scoring to the invasive region matters clinically
Running CD44 scoring over the whole tissue (including DCIS, stroma, and normal tissue) would produce a number that doesn't correspond to any clinically meaningful population of cells — it would be averaging in tissue types the marker isn't meant to characterize. Restricting scoring to the invasive tumor mask (from Stage J's segmentation) keeps the final number interpretable, which is precisely the design decision made in EPIC-009.

---

## Stage K — Machine Learning for Pathology

### 91. What is a Convolutional Neural Network (CNN) at a high level
A neural network architecture specialized for images: it learns small local filters (convolutions) that detect visual patterns (edges, textures, shapes) and stacks them into increasingly abstract features, ultimately learning to recognize complex patterns like "this looks like invasive tumor" directly from labeled examples — no one hand-writes the rule.

### 92. What is patch-based classification for whole slide images
Because whole slide images are too large to feed into a model at once, they're cut into small square "patches" (e.g. 256×256 pixels), each patch classified independently (e.g. "tumor" vs. "not tumor"), and the results stitched back into a full-slide map. This is the approach the codebase's dormant `PatchClassifierDetector` is designed around.

### 93. What is nuclei/cell segmentation with deep learning (Cellpose, StarDist)
Cellpose and StarDist are popular, pretrained deep-learning tools specifically for finding and outlining individual cells/nuclei in microscopy images — a much more precise alternative to threshold-based nucleus detection. The codebase includes exploratory scripts referencing Cellpose, but it isn't wired into the live pipeline yet.

### 94. What is transfer learning and why public datasets (BACH, TCGA) help
Transfer learning means starting from a model already trained on a large, related dataset, then fine-tuning it on your smaller target dataset — much more data-efficient than training from scratch. BACH (~400 labeled breast histology images) and TCGA-BRCA (a large public cancer genomics/imaging archive) are exactly the kind of public datasets that let a model learn general "what does breast tissue look like" patterns before ever seeing this project's client data.

### 95. What annotated training data looks like for pathology ML
Typically: the same slide image, plus a paired label — either a pixel-level mask (pathologist-drawn outline of the tumor region) for segmentation tasks, or a per-cell/per-patch category label ("positive"/"negative", "tumor"/"not tumor") for classification tasks. Producing this is slow, expert-time-intensive work, which is why it's treated as a scarce resource to request carefully (see Stage L).

### 96. Train/validation/test split and why sample size matters
Standard ML practice: split labeled data into a **training** set (model learns from this), a **validation** set (used to tune the model without touching test data), and a **test** set (final, untouched check of real-world performance). In pathology, sample size is doubly constrained — not just by ML convention, but by how few annotated slides a client can realistically provide, which is why this project treats "how many cases" as an open, carefully reasoned question rather than an assumed number.

---

## Stage L — Validation, Statistics & Regulatory

### 97. What is correlation and confidence interval in a validation study
**Correlation** measures how closely the algorithm's scores track the pathologist reference scores (e.g., Pearson's r, 0 to 1). A **confidence interval** expresses the uncertainty around that correlation estimate given the sample size — a small sample produces a wide interval, meaning the "true" correlation could plausibly be much lower (or higher) than what was observed.

### 98. Why small sample sizes give unreliable validation claims
With only a handful of cases (e.g. n=6, as in this project's initial internal batch), an observed correlation of 0.8 might have a 95% confidence interval spanning nearly 0 to 0.98 — so wide that you can't distinguish "this tool works great" from "this tool barely works at all." Small samples can reliably catch *gross* failure, but can't confirm success — which is exactly the documented rationale (in this project's open-questions register) for requesting ~25–30 cases before trusting any validation number.

### 99. Basics of software-as-a-medical-device (SaMD) regulatory concepts
Software that's intended to diagnose, treat, or inform clinical decisions (like an IHC scoring tool) is generally regulated as a medical device in its own right, separate from any hardware — requiring documented validation, risk classification, and often regulatory clearance (e.g. FDA, CE marking) before clinical use. This is part of why this codebase favors deterministic, explainable math over black-box ML by default — auditability is a regulatory asset, not just an engineering nicety.

### 100. Putting it together — how this project's pipeline maps to the pathologist's manual workflow
Every step in this app has a direct manual analogue:

| Manual pathologist step | This app's equivalent |
|---|---|
| Scan slide under microscope | Upload + whole-slide-image scan |
| Visually find invasive tumor region, exclude DCIS/stroma/fat | `HeuristicInvasiveDetector` — nuclear-density segmentation |
| Visually judge each cell's membrane staining (stained ring? how complete?) | `MembraneScorer` — DAB thresholding + ring-completeness check |
| Count % positive cells, judge average intensity | `percent_positive` / `mean_intensity` computation |
| Get 2–3 colleagues to agree (consensus scoring) | Pathologist validation against the algorithm's output (Stage L) |
| Write it into the pathology report | The score card / verified score you saw in the UI |

Nothing in the current pipeline replaces the pathologist's judgment — it's a faster, repeatable, auditable re-implementation of the same measurable steps a pathologist already performs by eye, validated against pathologists rather than instead of them.

---

*Next steps if you want to go deeper: read `backend/app/scoring/invasive.py` and `backend/app/scoring/membrane.py` side-by-side with this file — every function in those two files maps directly to one of the Stage H/I/J items above.*
