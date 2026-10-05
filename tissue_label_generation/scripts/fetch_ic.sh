#!/usr/bin/env bash
# The BRACS invasive-carcinoma regions of interest, two per patient.
#
# Approach 1 borrows in-situ epithelium from BRACS's DCIS regions because BCSS ships
# nine in-situ tiles in its entire release. The cost of that, recorded in
# `datasets.by_source_authority`, is that `source` became very nearly a synonym for
# `class 1` - 1,843 BRACS tiles against 9 from BCSS - so a model can score on in-situ by
# recognising which dataset it is looking at. `06_leakage_check.py` measures exactly that
# and reports INCONCLUSIVE.
#
# These regions are the fix: BRACS supplying invasive carcinoma as well means dataset
# identity no longer predicts the class. Their invasive label is BRACS's own
# three-pathologist consensus, not BEETLE's guess inside a DCIS region - which is what
# the 270 tiles v1 shipped and scored 0.19 Dice on actually were.
#
# 112 patients, 2 regions each -> 182 files, 5.5 GiB, against 649 and 15.4 GiB for the
# whole tree. See `fetch_roi.sh` for why the limit is applied at the listing.
set -uo pipefail
exec "$(dirname "$0")/fetch_roi.sh" 6_IC ic 2
