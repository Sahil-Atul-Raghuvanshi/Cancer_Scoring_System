#!/usr/bin/env bash
# The BRACS normal regions of interest, two per patient.
#
# **This is the download that targets the measured failure**, and the measurement is
# worth stating next to it. On `CAN_00251_26_H&E`, the v2 tile model called 45.9% of the
# section in-situ epithelium - 141 mm² - at a mean confidence of 0.809 and a median
# top-to-runner-up margin of 0.756. Confidently wrong, not hedging. For 88.5% of those
# windows the runner-up class was non-epithelium, so the in-situ head is firing on
# STROMA, and forcing a two-class decision moved invasive from 0.87% to 6.14%.
#
# The cause is that class 1 is 1,843 BRACS tiles against 9 from BCSS, so `source` and
# `class 1` are nearly the same question - `06_leakage_check.py` predicts source from the
# frozen features at 0.904 and reports INCONCLUSIVE. The in-situ head has learned "looks
# like a BRACS crop" and BRACS-like connective tissue on a third laboratory's slide
# clears it.
#
# A consensus-normal region is the correction, and it is the safest of the three trees:
# BRACS says the region contains no carcinoma, so BEETLE is only asked for the axis it is
# strong on - epithelium versus not - and never to adjudicate invasive against in-situ.
# Its stroma and fat become class 0 and its normal ducts class 1, both under a human
# consensus, which is what teaches "BRACS-looking stroma is not DCIS".
#
# 130 patients, 2 regions each -> 213 files, 2.6 GiB, against 484 and 4.4 GiB for the
# whole tree. See `fetch_roi.sh` for why the limit is applied at the listing.
set -uo pipefail
exec "$(dirname "$0")/fetch_roi.sh" 0_N normal 2
