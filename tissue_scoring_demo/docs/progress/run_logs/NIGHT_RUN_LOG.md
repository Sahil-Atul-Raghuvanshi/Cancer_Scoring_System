[2026-09-16 02:46:27] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-16 02:46:27]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-16 02:46:27]   scoring ['A', 'F', 'R', 'U', 'W']
[2026-09-16 02:46:27] 
[2026-09-16 02:46:27] ===== A (CD44) =====
[2026-09-16 02:46:27] A: reusing the pair already registered for this case
[2026-09-16 02:46:27] A: H&E W7nhinXLI_RAkpYK59xgfA  IHC FtkzunpsbePGb-0oc6gg6A
[2026-09-16 02:46:27] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 02:46:34] A: step 9 found 104 invasive region(s)
[2026-09-16 02:46:34] A: step 10, registering the two slides
[2026-09-16 02:46:34] A: step 11, segmenting nuclei
[2026-09-16 02:46:34] A: step 11 counted 2,435 nuclei over 5.166 mm2 (471/mm2)
[2026-09-16 02:46:34] A: step 12 typed 2,435 cells, 33.0% tumour
[2026-09-16 02:46:44] A: step 13 built 803 membrane compartments
[2026-09-16 02:46:54] A: step 14 measured 803 cells, mean DAB 0.2483 OD, mean ring_completeness 0.474
[2026-09-16 02:46:58] A: step 15 bins 0:369 / 1+:186 / 2+:214 / 3+:34
[2026-09-16 02:47:03] A: STEP 16 -> 65 % positive, intensity 1 (Weak)   [raw 64.15 % / 0.3871 OD]
[2026-09-16 02:47:03] A:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 02:47:03] A:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 02:47:03] A:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 64.2 %; pooling every measured cell regardless of which region it came from gives 53.5 %; a plain mean of the regions gives 27.6 %. Step 11 now allocates its fields in proportion to region area, so those first two figures are close where the tumour is one dominant mass and drift apart as the guaranteed minimum field per region lifts small regions above their share. The plain mean is the outlier by design: it gives a 0.25 mm2 fragment the same say as a 22 mm2 mass. The area-weighted figure is reported, which is the reading OncoStem's own procedure implies - the entire slide scanned, every field averaged (SOP 4.2) - though whether they weight by area or by cell count is still unanswered (Q3).
[2026-09-16 02:47:03] A: scored in 36s
[2026-09-16 02:47:03] 
[2026-09-16 02:47:03] ===== F (ABCC4 (MRP4)) =====
[2026-09-16 02:47:03] F: paired with the H&E already carrying steps 2-9 (W7nhinXLI_RAkpYK59xgfA), so step 8 does not run again
[2026-09-16 02:47:03] F: H&E W7nhinXLI_RAkpYK59xgfA  IHC T2VQb4rPUrHiC0aSwgl5ag
[2026-09-16 02:47:03] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 02:47:03] F: step 9 found 104 invasive region(s)
[2026-09-16 02:47:03] F: step 10, registering the two slides
[2026-09-16 02:56:56] F: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 02:56:56] F: step 11, segmenting nuclei
[2026-09-16 03:03:27] F: step 11 counted 3,890 nuclei over 5.166 mm2 (753/mm2)
[2026-09-16 03:03:27] F: step 12 typed 3,890 cells, 31.9% tumour
[2026-09-16 03:03:46] F: step 13 built 1,242 membrane compartments
[2026-09-16 03:03:52] F: step 14 measured 1,242 cells, mean DAB 0.4712 OD, mean ring_completeness 0.821
[2026-09-16 03:03:59] F: step 15 bins 0:43 / 1+:402 / 2+:586 / 3+:211
[2026-09-16 03:04:05] F: STEP 16 -> 95 % positive, intensity 1.5 (Moderate)   [raw 96.00 % / 0.4969 OD]
[2026-09-16 03:04:05] F:   caveat: OUTSIDE THE EXPECTED RANGE. ABCC4 (MRP4) has been reported between 35 % and 70 % across the cases OncoStem has read; this slide scores 95 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-16 03:04:05] F:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 03:04:05] F:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 62% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 03:04:05] F:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 03:04:05] F: scored in 1022s
[2026-09-16 03:04:05] 
[2026-09-16 03:04:05] ===== R (ABCC11 (MRP8)) =====
[2026-09-16 03:04:05] R: paired with the H&E already carrying steps 2-9 (W7nhinXLI_RAkpYK59xgfA), so step 8 does not run again
[2026-09-16 03:04:05] R: H&E W7nhinXLI_RAkpYK59xgfA  IHC D18Ilp81NfsONVqK_OxlxQ
[2026-09-16 03:04:05] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 03:04:05] R: step 9 found 104 invasive region(s)
[2026-09-16 03:04:05] R: step 10, registering the two slides
[2026-09-16 03:14:50] R: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 03:14:50] R: step 11, segmenting nuclei
[2026-09-16 03:20:33] R: step 11 counted 1,740 nuclei over 5.166 mm2 (337/mm2)
[2026-09-16 03:20:34] R: step 12 typed 1,740 cells, 38.0% tumour
[2026-09-16 03:20:48] R: step 13 built 662 membrane compartments
[2026-09-16 03:20:54] R: step 14 measured 662 cells, mean DAB 0.2220 OD, mean ring_completeness 0.507
[2026-09-16 03:21:01] R: step 15 bins 0:244 / 1+:314 / 2+:92 / 3+:12
[2026-09-16 03:21:08] R: STEP 16 -> 60 % positive, intensity 1 (Weak)   [raw 62.15 % / 0.2832 OD]
[2026-09-16 03:21:08] R:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 03:21:08] R:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 83% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 03:21:08] R:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 03:21:08] R:   caveat: Combining the invasive regions by area and averaging them plainly differ by 10.2 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-16 03:21:08] R: scored in 1023s
[2026-09-16 03:21:08] 
[2026-09-16 03:21:08] ===== U (N-cadherin (CDH2)) =====
[2026-09-16 03:21:08] U: paired with the H&E already carrying steps 2-9 (W7nhinXLI_RAkpYK59xgfA), so step 8 does not run again
[2026-09-16 03:21:08] U: H&E W7nhinXLI_RAkpYK59xgfA  IHC GNeN3Ld8idBhyWJsP9fcMQ
[2026-09-16 03:21:08] U: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 03:21:08] U: step 9 found 104 invasive region(s)
[2026-09-16 03:21:08] U: step 10, registering the two slides
[2026-09-16 03:33:45] U: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 03:33:45] U: step 11, segmenting nuclei
[2026-09-16 03:40:36] U: step 11 counted 2,344 nuclei over 5.166 mm2 (454/mm2)
[2026-09-16 03:40:36] U: step 12 typed 2,344 cells, 37.7% tumour
[2026-09-16 03:40:54] U: step 13 built 883 cytoplasm compartments
[2026-09-16 03:41:02] U: step 14 measured 883 cells, mean DAB 0.4315 OD, mean stained_fraction 0.797
[2026-09-16 03:41:10] U: step 15 bins 0:21 / 1+:219 / 2+:485 / 3+:158
[2026-09-16 03:41:18] U: STEP 16 -> 100 % positive, intensity 1.5 (Moderate)   [raw 97.65 % / 0.4408 OD]
[2026-09-16 03:41:18] U:   caveat: OUTSIDE THE EXPECTED RANGE. N-cadherin (CDH2) has been reported between 70 % and 85 % across the cases OncoStem has read; this slide scores 100 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-16 03:41:18] U:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 03:41:18] U:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 77% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 03:41:18] U:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 03:41:18] U: scored in 1210s
[2026-09-16 03:41:18] 
[2026-09-16 03:41:18] ===== W (Pan-cadherin) =====
[2026-09-16 03:41:18] W: paired with the H&E already carrying steps 2-9 (W7nhinXLI_RAkpYK59xgfA), so step 8 does not run again
[2026-09-16 03:41:18] W: H&E W7nhinXLI_RAkpYK59xgfA  IHC XESFnEPKmSukqBqt87NVpg
[2026-09-16 03:41:18] W: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 03:41:18] W: step 9 found 104 invasive region(s)
[2026-09-16 03:41:18] W: step 10, registering the two slides
[2026-09-16 03:54:06] W: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 03:54:06] W: step 11, segmenting nuclei
[2026-09-16 04:00:55] W: step 11 counted 2,497 nuclei over 5.166 mm2 (483/mm2)
[2026-09-16 04:00:56] W: step 12 typed 2,497 cells, 25.7% tumour
[2026-09-16 04:01:13] W: step 13 built 642 cytoplasm compartments
[2026-09-16 04:01:20] W: step 14 measured 642 cells, mean DAB 0.2877 OD, mean stained_fraction 0.655
[2026-09-16 04:01:28] W: step 15 bins 0:100 / 1+:287 / 2+:217 / 3+:38
[2026-09-16 04:01:35] W: STEP 16 -> 85 % positive, intensity 1 (Weak)   [raw 84.41 % / 0.3137 OD]
[2026-09-16 04:01:35] W:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 04:01:35] W:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 04:01:35] W:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 04:01:35] W:   caveat: Combining the invasive regions by area and averaging them plainly differ by 7.5 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-16 04:01:35] W: scored in 1217s
[2026-09-16 04:01:35] 
[2026-09-16 04:01:35] ===== summary =====
[2026-09-16 04:01:35]   A CD44                   scored                 65% / 1
[2026-09-16 04:01:35]   F ABCC4 (MRP4)           scored                 95% / 1.5
[2026-09-16 04:01:35]   R ABCC11 (MRP8)          scored                 60% / 1
[2026-09-16 04:01:35]   U N-cadherin (CDH2)      scored                 100% / 1.5
[2026-09-16 04:01:35]   W Pan-cadherin           scored                 85% / 1
[2026-09-16 04:04:35] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
[2026-09-16 04:21:11] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-16 04:21:11]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-16 04:21:11]   scoring ['A']
[2026-09-16 04:21:11] 
[2026-09-16 04:21:11] ===== A (CD44) =====
[2026-09-16 04:21:11] A: reusing the pair already registered for this case
[2026-09-16 04:21:11] A: H&E W7nhinXLI_RAkpYK59xgfA  IHC FtkzunpsbePGb-0oc6gg6A
[2026-09-16 04:21:11] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-16 04:21:21] A: step 9 found 104 invasive region(s)
[2026-09-16 04:21:21] A: step 10, registering the two slides
[2026-09-16 04:36:52] A: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-16 04:36:52] A: step 11, segmenting nuclei
[2026-09-16 04:44:48] A: step 11 counted 2,435 nuclei over 5.166 mm2 (471/mm2)
[2026-09-16 04:44:49] A: step 12 typed 2,435 cells, 33.0% tumour
[2026-09-16 04:45:09] A: step 13 built 803 membrane compartments
[2026-09-16 04:45:18] A: step 14 measured 803 cells, mean DAB 0.2483 OD, mean ring_completeness 0.474
[2026-09-16 04:45:27] A: step 15 bins 0:369 / 1+:186 / 2+:214 / 3+:34
[2026-09-16 04:45:36] A: STEP 16 -> 65 % positive, intensity 1 (Weak)   [raw 64.15 % / 0.3871 OD]
[2026-09-16 04:45:36] A:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-16 04:45:36] A:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-16 04:45:36] A:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-16 04:45:36] A:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 64.2 %; pooling every measured cell regardless of which region it came from gives 53.5 %; a plain mean of the regions gives 27.6 %. Step 11 now allocates its fields in proportion to region area, so those first two figures are close where the tumour is one dominant mass and drift apart as the guaranteed minimum field per region lifts small regions above their share. The plain mean is the outlier by design: it gives a 0.25 mm2 fragment the same say as a 22 mm2 mass. The area-weighted figure is reported, which is the reading OncoStem's own procedure implies - the entire slide scanned, every field averaged (SOP 4.2) - though whether they weight by area or by cell count is still unanswered (Q3).
[2026-09-16 04:45:36] A: scored in 1465s
[2026-09-16 04:45:36] 
[2026-09-16 04:45:36] ===== summary =====
[2026-09-16 04:45:36]   A CD44                   scored                 65% / 1
[2026-09-16 04:46:06] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_A_rerun.xlsx
