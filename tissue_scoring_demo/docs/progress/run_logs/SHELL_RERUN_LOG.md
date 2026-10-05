[2026-09-16 10:11:50] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-16 10:11:50]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-16 10:11:50]   scoring ['A', 'F', 'R']
[2026-09-16 10:11:50] 
[2026-09-16 10:11:50] ===== A (CD44) =====
[2026-09-16 10:11:50] A: reusing the pair already registered for this case
[2026-09-16 10:11:50] A: H&E W7nhinXLI_RAkpYK59xgfA  IHC FtkzunpsbePGb-0oc6gg6A
[2026-09-16 10:11:50] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:11:55] A: step 9 found 104 invasive region(s)
[2026-09-16 10:11:55] A: step 10, registering the two slides
[2026-09-16 10:11:55] A: step 11, segmenting nuclei
[2026-09-16 10:11:55] A: step 11 counted 2,435 nuclei over 5.166 mm2 (471/mm2)
[2026-09-16 10:11:56] A: step 12 typed 2,435 cells, 33.0% tumour
[2026-09-16 10:11:56] A: FAILED during step 13 (compartments) - NameError: name 'shell' is not defined
[2026-09-16 10:11:56] Traceback (most recent call last):
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\scripts\run_case_scores.py", line 246, in run_marker
    compartments = compartment_service.report(
        session.he_upload_id, session.ihc_upload_id
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 330, in report
    cached = self._cached(
        he_upload_id,
    ...<20 lines>...
        ],
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 238, in _cached
    expected_ring = shell
                    ^^^^^
NameError: name 'shell' is not defined. Did you mean: 'self'?

[2026-09-16 10:11:56] A: failed in 5s
[2026-09-16 10:11:56] 
[2026-09-16 10:11:56] ===== F (ABCC4 (MRP4)) =====
[2026-09-16 10:11:56] F: reusing the pair already registered for this case
[2026-09-16 10:11:56] F: H&E W7nhinXLI_RAkpYK59xgfA  IHC T2VQb4rPUrHiC0aSwgl5ag
[2026-09-16 10:11:56] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:11:56] F: step 9 found 104 invasive region(s)
[2026-09-16 10:11:56] F: step 10, registering the two slides
[2026-09-16 10:13:25] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-16 10:13:25]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-16 10:13:25]   scoring ['A', 'F', 'R']
[2026-09-16 10:13:25] 
[2026-09-16 10:13:25] ===== A (CD44) =====
[2026-09-16 10:13:25] A: reusing the pair already registered for this case
[2026-09-16 10:13:25] A: H&E W7nhinXLI_RAkpYK59xgfA  IHC FtkzunpsbePGb-0oc6gg6A
[2026-09-16 10:13:25] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:13:39] A: step 9 found 104 invasive region(s)
[2026-09-16 10:13:39] A: step 10, registering the two slides
[2026-09-16 10:13:39] A: step 11, segmenting nuclei
[2026-09-16 10:13:39] A: step 11 counted 2,435 nuclei over 5.166 mm2 (471/mm2)
[2026-09-16 10:13:40] A: step 12 typed 2,435 cells, 33.0% tumour
[2026-09-16 10:13:40] A: FAILED during step 13 (compartments) - NameError: name 'shell' is not defined
[2026-09-16 10:13:40] Traceback (most recent call last):
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\scripts\run_case_scores.py", line 246, in run_marker
    compartments = compartment_service.report(
        session.he_upload_id, session.ihc_upload_id
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 330, in report
    cached = self._cached(
        he_upload_id,
    ...<20 lines>...
        ],
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 238, in _cached
    expected_ring = shell
                    ^^^^^
NameError: name 'shell' is not defined. Did you mean: 'self'?

[2026-09-16 10:13:40] A: failed in 15s
[2026-09-16 10:13:40] 
[2026-09-16 10:13:40] ===== F (ABCC4 (MRP4)) =====
[2026-09-16 10:13:40] F: reusing the pair already registered for this case
[2026-09-16 10:13:40] F: H&E W7nhinXLI_RAkpYK59xgfA  IHC T2VQb4rPUrHiC0aSwgl5ag
[2026-09-16 10:13:40] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:13:40] F: step 9 found 104 invasive region(s)
[2026-09-16 10:13:40] F: step 10, registering the two slides
[2026-09-16 10:14:53] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-16 10:14:53]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-16 10:14:53]   scoring ['A', 'F', 'R']
[2026-09-16 10:14:53] 
[2026-09-16 10:14:53] ===== A (CD44) =====
[2026-09-16 10:14:53] A: reusing the pair already registered for this case
[2026-09-16 10:14:53] A: H&E W7nhinXLI_RAkpYK59xgfA  IHC FtkzunpsbePGb-0oc6gg6A
[2026-09-16 10:14:53] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:15:12] A: step 9 found 104 invasive region(s)
[2026-09-16 10:15:12] A: step 10, registering the two slides
[2026-09-16 10:15:12] A: step 11, segmenting nuclei
[2026-09-16 10:15:12] A: step 11 counted 2,435 nuclei over 5.166 mm2 (471/mm2)
[2026-09-16 10:15:12] A: step 12 typed 2,435 cells, 33.0% tumour
[2026-09-16 10:16:11] A: step 13 built 803 membrane compartments
[2026-09-16 10:16:53] A: step 14 measured 803 cells, mean DAB 0.2253 OD, mean ring_completeness 0.415
[2026-09-16 10:17:08] A: step 15 bins 0:393 / 1+:192 / 2+:191 / 3+:27
[2026-09-16 10:17:19] A: STEP 16 -> 60 % positive, intensity 1 (Weak)   [raw 59.77 % / 0.3746 OD]
[2026-09-16 10:17:19] A:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 10:17:19] A:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 10:17:19] A:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 10:17:19] A:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 59.8 %; pooling every measured cell regardless of which region it came from gives 49.7 %; a plain mean of the regions gives 23.1 %. Step 11 now allocates its fields in proportion to region area, so those first two figures are close where the tumour is one dominant mass and drift apart as the guaranteed minimum field per region lifts small regions above their share. The plain mean is the outlier by design: it gives a 0.25 mm2 fragment the same say as a 22 mm2 mass. The area-weighted figure is reported, which is the reading OncoStem's own procedure implies - the entire slide scanned, every field averaged (SOP 4.2) - though whether they weight by area or by cell count is still unanswered (Q3).
[2026-09-16 10:17:19] A: scored in 146s
[2026-09-16 10:17:19] 
[2026-09-16 10:17:19] ===== F (ABCC4 (MRP4)) =====
[2026-09-16 10:17:19] F: reusing the pair already registered for this case
[2026-09-16 10:17:19] F: H&E W7nhinXLI_RAkpYK59xgfA  IHC T2VQb4rPUrHiC0aSwgl5ag
[2026-09-16 10:17:19] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:17:19] F: step 9 found 104 invasive region(s)
[2026-09-16 10:17:19] F: step 10, registering the two slides
[2026-09-16 10:20:00] F: step 10 refused - only 0 matched features, need at least 50 - the two sections have too little in common to fit a transform to; no residual error was recorded, so how far apart the matched features ended up is unknown and cannot be checked (VALIS recorded no registration summary, and none was found on disk either. Its output directory may be too deep for the filesystem to write into - check that <dst_dir>/<name>/data/ exists.)
[2026-09-16 10:20:00] F: alignment_refused in 161s
[2026-09-16 10:20:00] 
[2026-09-16 10:20:00] ===== R (ABCC11 (MRP8)) =====
[2026-09-16 10:20:00] R: reusing the pair already registered for this case
[2026-09-16 10:20:00] R: H&E W7nhinXLI_RAkpYK59xgfA  IHC D18Ilp81NfsONVqK_OxlxQ
[2026-09-16 10:20:00] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:20:00] R: step 9 found 104 invasive region(s)
[2026-09-16 10:20:00] R: step 10, registering the two slides
[2026-09-16 10:20:00] R: step 11, segmenting nuclei
[2026-09-16 10:20:00] R: step 11 counted 1,740 nuclei over 5.166 mm2 (337/mm2)
[2026-09-16 10:20:01] R: step 12 typed 1,740 cells, 38.0% tumour
[2026-09-16 10:20:32] R: step 13 built 662 membrane compartments
[2026-09-16 10:20:54] R: step 14 measured 662 cells, mean DAB 0.2064 OD, mean ring_completeness 0.440
[2026-09-16 10:21:05] R: step 15 bins 0:279 / 1+:295 / 2+:78 / 3+:10
[2026-09-16 10:21:15] R: STEP 16 -> 55 % positive, intensity 1 (Weak)   [raw 54.04 % / 0.2805 OD]
[2026-09-16 10:21:15] R:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 10:21:15] R:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 83% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 10:21:15] R:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 10:21:15] R:   caveat: Combining the invasive regions by area and averaging them plainly differ by 5.6 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-16 10:21:15] R: scored in 75s
[2026-09-16 10:21:15] 
[2026-09-16 10:21:15] ===== summary =====
[2026-09-16 10:21:15]   A CD44                   scored                 60% / 1
[2026-09-16 10:21:15]   F ABCC4 (MRP4)           alignment_refused      -
[2026-09-16 10:21:15]   R ABCC11 (MRP8)          scored                 55% / 1
[2026-09-16 10:22:50] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
[2026-09-16 10:23:23] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-16 10:23:23]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-16 10:23:23]   scoring ['F']
[2026-09-16 10:23:23] 
[2026-09-16 10:23:23] ===== F (ABCC4 (MRP4)) =====
[2026-09-16 10:23:23] F: reusing the pair already registered for this case
[2026-09-16 10:23:23] F: H&E W7nhinXLI_RAkpYK59xgfA  IHC T2VQb4rPUrHiC0aSwgl5ag
[2026-09-16 10:23:23] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:23:32] F: step 9 found 104 invasive region(s)
[2026-09-16 10:23:32] F: step 10, registering the two slides
[2026-09-16 10:26:02] F: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 10:26:02] F: step 11, segmenting nuclei
[2026-09-16 10:28:21] F: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 10:28:21] F: step 11, segmenting nuclei
[2026-09-16 10:29:44] F: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 10:29:44] F: step 11, segmenting nuclei
[2026-09-16 10:34:39] F: step 11 counted 3,890 nuclei over 5.166 mm2 (753/mm2)
[2026-09-16 10:34:40] F: step 12 typed 3,890 cells, 31.9% tumour
[2026-09-16 10:35:33] F: FAILED during step 13 (compartments) - NameError: name 'shell' is not defined
[2026-09-16 10:35:33] Traceback (most recent call last):
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\scripts\run_case_scores.py", line 246, in run_marker
    compartments = compartment_service.report(
        session.he_upload_id, session.ihc_upload_id
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 546, in report
    width_sensitivity=self._sweep(
                      ^^^^^^^^^^^^
        he_upload_id,
        ^^^^^^^^^^^^^
        ihc_upload_id,
    
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 603, in _sweep
    keep = self._tumour_ids(
    
NameError: name 'shell' is not defined. Did you mean: 'self'?

[2026-09-16 10:35:33] F: failed in 1418s
[2026-09-16 10:35:33] 
[2026-09-16 10:35:33] ===== R (ABCC11 (MRP8)) =====
[2026-09-16 10:35:33] R: reusing the pair already registered for this case
[2026-09-16 10:35:33] R: H&E W7nhinXLI_RAkpYK59xgfA  IHC D18Ilp81NfsONVqK_OxlxQ
[2026-09-16 10:35:33] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:35:34] R: step 9 found 104 invasive region(s)
[2026-09-16 10:35:34] R: step 10, registering the two slides
[2026-09-16 10:35:34] R: step 11, segmenting nuclei
[2026-09-16 10:35:34] R: step 11 counted 1,740 nuclei over 5.166 mm2 (337/mm2)
[2026-09-16 10:35:34] R: step 12 typed 1,740 cells, 38.0% tumour
[2026-09-16 10:35:34] R: FAILED during step 13 (compartments) - NameError: name 'shell' is not defined
[2026-09-16 10:35:34] Traceback (most recent call last):
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\scripts\run_case_scores.py", line 246, in run_marker
    compartments = compartment_service.report(
        session.he_upload_id, session.ihc_upload_id
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 330, in report
    cached = self._cached(
        he_upload_id,
    ...<20 lines>...
        ],
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 238, in _cached
    expected_ring = shell_um
                    ^^^^^
NameError: name 'shell' is not defined. Did you mean: 'self'?

[2026-09-16 10:35:34] R: failed in 1s
[2026-09-16 10:35:34] 
[2026-09-16 10:35:34] ===== summary =====
[2026-09-16 10:35:34]   A CD44                   failed                 -
[2026-09-16 10:35:34]   F ABCC4 (MRP4)           failed                 -
[2026-09-16 10:35:34]   R ABCC11 (MRP8)          failed                 -
[2026-09-16 10:39:39] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
[2026-09-16 10:41:02] F: step 11 counted 3,890 nuclei over 5.166 mm2 (753/mm2)
[2026-09-16 10:41:03] F: step 12 typed 3,890 cells, 31.9% tumour
[2026-09-16 10:42:30] F: step 13 built 1,242 membrane compartments
[2026-09-16 10:42:49] F: step 14 measured 1,242 cells, mean DAB 0.4430 OD, mean ring_completeness 0.710
[2026-09-16 10:43:08] F: step 15 bins 0:69 / 1+:458 / 2+:523 / 3+:192
[2026-09-16 10:43:21] F: step 11 counted 3,890 nuclei over 5.166 mm2 (753/mm2)
[2026-09-16 10:43:21] F: step 12 typed 3,890 cells, 31.9% tumour
[2026-09-16 10:43:23] F: STEP 16 -> 90 % positive, intensity 1.5 (Moderate)   [raw 92.35 % / 0.4790 OD]
[2026-09-16 10:43:23] F:   caveat: OUTSIDE THE EXPECTED RANGE. ABCC4 (MRP4) has been reported between 35 % and 70 % across the cases OncoStem has read; this slide scores 90 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-16 10:43:23] F:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 10:43:23] F:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 62% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 10:43:23] F:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 10:43:23] F:   caveat: Combining the invasive regions by area and averaging them plainly differ by 6.5 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-16 10:43:23] F: scored in 1784s
[2026-09-16 10:43:23] 
[2026-09-16 10:43:23] ===== R (ABCC11 (MRP8)) =====
[2026-09-16 10:43:23] R: reusing the pair already registered for this case
[2026-09-16 10:43:23] R: H&E W7nhinXLI_RAkpYK59xgfA  IHC D18Ilp81NfsONVqK_OxlxQ
[2026-09-16 10:43:23] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 10:43:24] R: step 9 found 104 invasive region(s)
[2026-09-16 10:43:24] R: step 10, registering the two slides
[2026-09-16 10:43:24] R: step 11, segmenting nuclei
[2026-09-16 10:43:24] R: step 11 counted 1,740 nuclei over 5.166 mm2 (337/mm2)
[2026-09-16 10:43:24] R: step 12 typed 1,740 cells, 38.0% tumour
[2026-09-16 10:43:24] R: FAILED during step 13 (compartments) - NameError: name 'shell' is not defined
[2026-09-16 10:43:24] Traceback (most recent call last):
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\scripts\run_case_scores.py", line 246, in run_marker
    compartments = compartment_service.report(
        session.he_upload_id, session.ihc_upload_id
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 330, in report
    cached = self._cached(
        he_upload_id,
    ...<20 lines>...
        ],
    )
  File "C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\tissue_scoring_demo\backend\app\services\compartment_service.py", line 238, in _cached
    expected_ring = shell_um
                    ^^^^^
NameError: name 'shell' is not defined. Did you mean: 'self'?

[2026-09-16 10:43:24] R: failed in 0s
[2026-09-16 10:43:24] 
[2026-09-16 10:43:24] ===== summary =====
[2026-09-16 10:43:24]   A CD44                   failed                 -
[2026-09-16 10:43:24]   F ABCC4 (MRP4)           scored                 90% / 1.5
[2026-09-16 10:43:24]   R ABCC11 (MRP8)          failed                 -
[2026-09-16 10:44:00] F: step 13 built 1,242 membrane compartments
[2026-09-16 10:44:11] F: step 14 measured 1,242 cells, mean DAB 0.4430 OD, mean ring_completeness 0.710
[2026-09-16 10:44:22] F: step 15 bins 0:69 / 1+:458 / 2+:523 / 3+:192
[2026-09-16 10:44:33] F: STEP 16 -> 90 % positive, intensity 1.5 (Moderate)   [raw 92.35 % / 0.4790 OD]
[2026-09-16 10:44:33] F:   caveat: OUTSIDE THE EXPECTED RANGE. ABCC4 (MRP4) has been reported between 35 % and 70 % across the cases OncoStem has read; this slide scores 90 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-16 10:44:33] F:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 10:44:33] F:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 62% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 10:44:33] F:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 10:44:33] F:   caveat: Combining the invasive regions by area and averaging them plainly differ by 6.5 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-16 10:44:33] F: scored in 1270s
[2026-09-16 10:44:33] 
[2026-09-16 10:44:33] ===== summary =====
[2026-09-16 10:44:33]   F ABCC4 (MRP4)           scored                 90% / 1.5
[2026-09-16 10:45:41] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
[2026-09-16 10:46:31] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_F_recover.xlsx
