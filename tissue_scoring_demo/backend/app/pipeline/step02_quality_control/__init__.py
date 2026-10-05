"""Step 2 - quality control.

Two independent things live in this package, and the split is the whole point:

  GrandQC   a pair of pre-trained segmentation models that decide *what* is
            wrong with the slide. Tissue detection first (cheap, 10 um/px),
            then multi-class artefact segmentation over the tissue it found.
            Accurate, and a black box.

  features  classical, per-tile image statistics - sharpness, contrast,
            texture - that explain *why* a region looks wrong. Transparent,
            and far too naive to be trusted as the decision.

The pipeline guide puts it plainly: use GrandQC for the result, classical
feature maps for the explanation. Neither is asked to do the other's job.

--------------------------------------------------------------------------
GrandQC attribution
--------------------------------------------------------------------------
The artefact and tissue models are GrandQC, and are used under GrandQC's
own non-commercial licence - see the licence shipped with the checkpoints.
This project reimplements the inference loop against its own slide reader;
the model weights and the method are GrandQC's.

  Weng Z. et al. "GrandQC: a comprehensive solution to quality control
  problem in digital pathology". Nature Communications (2024).
  https://doi.org/10.1038/s41467-024-54769-y
"""

from app.pipeline.step02_quality_control.classes import (
    ARTEFACT_CLASSES,
    BACKGROUND,
    QC_CLASSES,
    TISSUE,
    QCClass,
    class_by_id,
)

__all__ = [
    "ARTEFACT_CLASSES",
    "BACKGROUND",
    "QC_CLASSES",
    "TISSUE",
    "QCClass",
    "class_by_id",
]
