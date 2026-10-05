[2026-09-15 11:08:01] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-15 11:08:01]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-15 11:08:01]   scoring ['A', 'F', 'R', 'U', 'W']
[2026-09-15 11:08:01] 
[2026-09-15 11:08:01] ===== A (CD44) =====
[2026-09-15 11:08:01] A: reusing the pair already registered for this case
[2026-09-15 11:08:01] A: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC _YAC0Eaf5Hq-hlefyT362g
[2026-09-15 11:08:01] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 11:08:05] A: step 9 found 104 invasive region(s)
[2026-09-15 11:08:05] A: step 10, registering the two slides
[2026-09-15 11:11:20] A: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 11:11:20] A: step 11, segmenting nuclei
[2026-09-15 11:17:09] A: step 11 counted 2,435 nuclei over 5.166 mm2 (471/mm2)
[2026-09-15 11:17:09] A: step 12 typed 2,435 cells, 33.0% tumour
[2026-09-15 11:17:12] A: step 13 built 803 membrane compartments
[2026-09-15 11:17:22] A: step 14 measured 803 cells, mean DAB 0.2483 OD, mean ring_completeness 0.474
[2026-09-15 11:17:27] A: step 15 bins 0:369 / 1+:186 / 2+:214 / 3+:34
[2026-09-15 11:17:32] A: STEP 16 -> 65 % positive, intensity 1 (Weak)   [raw 64.15 % / 0.3871 OD]
[2026-09-15 11:17:32] A:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 11:17:32] A:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 11:17:32] A:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 11:17:32] A:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 64.2 %; pooling every measured cell regardless of which region it came from gives 53.5 %; a plain mean of the regions gives 27.6 %. Step 11 samples a fixed number of fields per region however large the region is, so pooling weights a small region as heavily as a large one. The area-weighted figure is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 11:17:32] A: scored in 572s
[2026-09-15 11:17:32] 
[2026-09-15 11:17:32] ===== F (ABCC4 (MRP4)) =====
[2026-09-15 11:17:32] F: reusing the pair already registered for this case
[2026-09-15 11:17:32] F: H&E rLl5AdGX41FVsd8tojLhiw  IHC ZVr5wT6LHe_VyOsesxk9Sg
[2026-09-15 11:17:32] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 11:17:35] F: step 9 found 104 invasive region(s)
[2026-09-15 11:17:35] F: step 10, registering the two slides
[2026-09-15 11:20:53] F: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 11:20:53] F: step 11, segmenting nuclei
[2026-09-15 11:24:57] F: step 11 counted 3,890 nuclei over 5.166 mm2 (753/mm2)
[2026-09-15 11:24:58] F: step 12 typed 3,890 cells, 31.9% tumour
[2026-09-15 11:25:02] F: step 13 built 1,242 membrane compartments
[2026-09-15 11:25:12] F: step 14 measured 1,242 cells, mean DAB 0.4712 OD, mean ring_completeness 0.821
[2026-09-15 11:25:17] F: step 15 bins 0:43 / 1+:402 / 2+:586 / 3+:211
[2026-09-15 11:25:21] F: STEP 16 -> 95 % positive, intensity 1.5 (Moderate)   [raw 96.00 % / 0.4969 OD]
[2026-09-15 11:25:21] F:   caveat: OUTSIDE THE EXPECTED RANGE. ABCC4 (MRP4) has been reported between 35 % and 70 % across the cases OncoStem has read; this slide scores 95 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-15 11:25:21] F:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 11:25:21] F:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 62% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 11:25:21] F:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 11:25:21] F: scored in 469s
[2026-09-15 11:25:21] 
[2026-09-15 11:25:21] ===== R (ABCC11 (MRP8)) =====
[2026-09-15 11:25:21] R: reusing the pair already registered for this case
[2026-09-15 11:25:21] R: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC WIxXB152BDd6-HTQVQmLQw
[2026-09-15 11:25:21] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 11:25:23] R: step 9 found 104 invasive region(s)
[2026-09-15 11:25:23] R: step 10, registering the two slides
[2026-09-15 11:28:20] R: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 11:28:20] R: step 11, segmenting nuclei
[2026-09-15 11:32:38] R: step 11 counted 1,740 nuclei over 5.166 mm2 (337/mm2)
[2026-09-15 11:32:38] R: step 12 typed 1,740 cells, 38.0% tumour
[2026-09-15 11:32:42] R: step 13 built 662 membrane compartments
[2026-09-15 11:32:53] R: step 14 measured 662 cells, mean DAB 0.2220 OD, mean ring_completeness 0.507
[2026-09-15 11:32:57] R: step 15 bins 0:244 / 1+:314 / 2+:92 / 3+:12
[2026-09-15 11:33:02] R: STEP 16 -> 60 % positive, intensity 1 (Weak)   [raw 62.15 % / 0.2832 OD]
[2026-09-15 11:33:02] R:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 11:33:02] R:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 83% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 11:33:02] R:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 11:33:02] R:   caveat: Combining the invasive regions by area and averaging them plainly differ by 10.2 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 11:33:02] R: scored in 461s
[2026-09-15 11:33:02] 
[2026-09-15 11:33:02] ===== U (N-cadherin (CDH2)) =====
[2026-09-15 11:33:02] U: reusing the pair already registered for this case
[2026-09-15 11:33:02] U: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC _hwDSP734Nx1FddsedpWhA
[2026-09-15 11:33:02] U: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 11:33:03] U: step 9 found 104 invasive region(s)
[2026-09-15 11:33:03] U: step 10, registering the two slides
[2026-09-15 11:36:15] U: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 11:36:15] U: step 11, segmenting nuclei
[2026-09-15 11:41:39] U: step 11 counted 2,344 nuclei over 5.166 mm2 (454/mm2)
[2026-09-15 11:41:39] U: step 12 typed 2,344 cells, 37.7% tumour
[2026-09-15 11:41:46] U: step 13 built 883 cytoplasm compartments
[2026-09-15 11:42:06] U: step 14 measured 883 cells, mean DAB 0.4315 OD, mean stained_fraction 0.797
[2026-09-15 11:42:15] U: step 15 bins 0:21 / 1+:219 / 2+:485 / 3+:158
[2026-09-15 11:42:22] U: STEP 16 -> 100 % positive, intensity 1.5 (Moderate)   [raw 97.65 % / 0.4408 OD]
[2026-09-15 11:42:22] U:   caveat: OUTSIDE THE EXPECTED RANGE. N-cadherin (CDH2) has been reported between 70 % and 85 % across the cases OncoStem has read; this slide scores 100 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-15 11:42:22] U:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 11:42:22] U:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 77% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 11:42:22] U:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 11:42:22] U: scored in 560s
[2026-09-15 11:42:22] 
[2026-09-15 11:42:22] ===== W (Pan-cadherin) =====
[2026-09-15 11:42:22] W: reusing the pair already registered for this case
[2026-09-15 11:42:22] W: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC TiBGQDku0gwM0qjwHUldFQ
[2026-09-15 11:42:22] W: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 11:42:24] W: step 9 found 104 invasive region(s)
[2026-09-15 11:42:24] W: step 10, registering the two slides
[2026-09-15 11:45:31] W: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 11:45:31] W: step 11, segmenting nuclei
[2026-09-15 11:50:02] W: step 11 counted 2,497 nuclei over 5.166 mm2 (483/mm2)
[2026-09-15 11:50:02] W: step 12 typed 2,497 cells, 25.7% tumour
[2026-09-15 11:50:05] W: step 13 built 642 cytoplasm compartments
[2026-09-15 11:50:15] W: step 14 measured 642 cells, mean DAB 0.2877 OD, mean stained_fraction 0.655
[2026-09-15 11:50:21] W: step 15 bins 0:100 / 1+:287 / 2+:217 / 3+:38
[2026-09-15 11:50:27] W: STEP 16 -> 85 % positive, intensity 1 (Weak)   [raw 84.41 % / 0.3137 OD]
[2026-09-15 11:50:27] W:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 11:50:27] W:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 11:50:27] W:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 11:50:27] W:   caveat: Combining the invasive regions by area and averaging them plainly differ by 7.5 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 11:50:27] W: scored in 484s
[2026-09-15 11:50:27] 
[2026-09-15 11:50:27] ===== summary =====
[2026-09-15 11:50:27]   A CD44                   scored                 65% / 1
[2026-09-15 11:50:27]   F ABCC4 (MRP4)           scored                 95% / 1.5
[2026-09-15 11:50:27]   R ABCC11 (MRP8)          scored                 60% / 1
[2026-09-15 11:50:27]   U N-cadherin (CDH2)      scored                 100% / 1.5
[2026-09-15 11:50:27]   W Pan-cadherin           scored                 85% / 1
[2026-09-15 11:52:50] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
