# Docs

The latest scores and the plain-language walkthrough are not here. They are in
[`results/`](../../results/) at the repository root (`HOW-IT-WORKS.md`, `DECISIONS.md`
and the score CSVs).

| Folder | What is in it |
| --- | --- |
| [`guides/`](guides/) | How the pipeline works and how to read it: the step-by-step demo guide, the pathology primer, the image-to-score data contract, step-by-step explainers |
| [`design/`](design/) | Build plans and checklists for individual features: five markers, step 7 branches, steps 10-13, Fix 1 + Fix 2, foundation-model features |
| [`segmentation_research/`](segmentation_research/) | The tissue-type segmentation study: every approach considered, the training plans, and their results |
| [`registration/`](registration/) | H&E-to-IHC stack registration: the full plan with every measurement, and the operator's resume notes |
| [`progress/`](progress/) | Dated progress logs and session checkpoints, plus `run_logs/` from the unattended runs. Historical, not maintained |
| [`reference/`](reference/) | The source research PDFs (gitignored) |

## Where to start

1. [`guides/HOW_THE_PIPELINE_WORKS.md`](guides/HOW_THE_PIPELINE_WORKS.md): segmentation end to end
2. [`guides/demo-pipeline-guide.md`](guides/demo-pipeline-guide.md): every step of the demo
3. [`guides/images-to-scores-mapping.md`](guides/images-to-scores-mapping.md): what data goes in and which score comes out
4. [`registration/REGISTRATION-PLAN.md`](registration/REGISTRATION-PLAN.md): how the IHC slides are aligned to the H&E
