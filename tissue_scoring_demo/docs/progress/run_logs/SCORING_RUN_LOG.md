[2026-09-15 03:17:53] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-15 03:17:53]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-15 03:17:53]   scoring ['A', 'F', 'R', 'U', 'W']
[2026-09-15 03:17:53] 
[2026-09-15 03:17:53] ===== A (CD44) =====
[2026-09-15 03:17:53] A: reusing the pair already registered for this case
[2026-09-15 03:17:53] A: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC _YAC0Eaf5Hq-hlefyT362g
[2026-09-15 03:17:53] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 03:17:58] A: step 9 found 104 invasive region(s)
[2026-09-15 03:17:58] A: step 10, registering the two slides
[2026-09-15 03:17:58] A: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 03:17:58] A: step 11, segmenting nuclei
[2026-09-15 03:17:58] A: step 11 counted 1,064 nuclei over 1.937 mm2 (549/mm2)
[2026-09-15 03:17:58] A: step 12 typed 1,064 cells, 32.7% tumour
[2026-09-15 03:18:00] A: step 13 built 965 membrane compartments
[2026-09-15 03:18:06] A: step 14 measured 965 cells, mean DAB 0.2618 OD, mean ring_completeness 0.518
[2026-09-15 03:18:07] A: step 15 bins 0:388 / 1+:304 / 2+:221 / 3+:52
[2026-09-15 03:18:09] A: STEP 16 -> 80 % positive, intensity 1.5 (Moderate)   [raw 80.17 % / 0.4323 OD]
[2026-09-15 03:18:09] A:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 03:18:09] A:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 72% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 03:18:09] A:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 03:18:09] A:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 80.2 %; pooling every measured cell regardless of which region it came from gives 59.3 %; a plain mean of the regions gives 55.2 %. Step 11 samples a fixed number of fields per region however large the region is, so pooling weights a small region as heavily as a large one. The area-weighted figure is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 03:18:09] A: scored in 16s
[2026-09-15 03:18:09] 
[2026-09-15 03:18:09] ===== F (ABCC4 (MRP4)) =====
[2026-09-15 03:18:09] F: reusing the pair already registered for this case
[2026-09-15 03:18:09] F: H&E rLl5AdGX41FVsd8tojLhiw  IHC ZVr5wT6LHe_VyOsesxk9Sg
[2026-09-15 03:18:09] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 03:18:11] F: step 9 found 104 invasive region(s)
[2026-09-15 03:18:11] F: step 10, registering the two slides
[2026-09-15 03:18:11] F: step 11, segmenting nuclei
[2026-09-15 03:19:50] F: step 11 counted 1,244 nuclei over 1.937 mm2 (642/mm2)
[2026-09-15 03:19:50] F: step 12 typed 1,244 cells, 32.0% tumour
[2026-09-15 03:19:52] F: step 13 built 1,314 membrane compartments
[2026-09-15 03:19:58] F: step 14 measured 1,314 cells, mean DAB 0.5517 OD, mean ring_completeness 0.818
[2026-09-15 03:20:00] F: step 15 bins 0:66 / 1+:371 / 2+:484 / 3+:393
[2026-09-15 03:20:03] F: STEP 16 -> 95 % positive, intensity 1.5 (Moderate)   [raw 97.47 % / 0.5694 OD]
[2026-09-15 03:20:03] F:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 03:20:03] F:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 68% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 03:20:03] F: scored in 114s
[2026-09-15 03:20:03] 
[2026-09-15 03:20:03] ===== R (ABCC11 (MRP8)) =====
[2026-09-15 03:20:03] R: paired with the H&E already carrying steps 2-9 (4XWhs5vekJ97Rx3_sflVOQ), so step 8 does not run again
[2026-09-15 03:20:03] R: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC WIxXB152BDd6-HTQVQmLQw
[2026-09-15 03:20:03] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 03:20:04] R: step 9 found 104 invasive region(s)
[2026-09-15 03:20:04] R: step 10, registering the two slides
[2026-09-15 03:29:06] R: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 03:29:06] R: step 11, segmenting nuclei
[2026-09-15 03:31:14] R: step 11 counted 888 nuclei over 1.937 mm2 (458/mm2)
[2026-09-15 03:31:14] R: step 12 typed 888 cells, 31.6% tumour
[2026-09-15 03:31:16] R: step 13 built 1,134 membrane compartments
[2026-09-15 03:31:24] R: step 14 measured 1,134 cells, mean DAB 0.3310 OD, mean ring_completeness 0.739
[2026-09-15 03:31:26] R: step 15 bins 0:165 / 1+:504 / 2+:412 / 3+:53
[2026-09-15 03:31:29] R: STEP 16 -> 75 % positive, intensity 1 (Weak)   [raw 74.25 % / 0.3135 OD]
[2026-09-15 03:31:29] R:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 03:31:29] R:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 77% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 03:31:29] R:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 03:31:29] R:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 74.2 %; pooling every measured cell regardless of which region it came from gives 84.5 %; a plain mean of the regions gives 82.3 %. Step 11 samples a fixed number of fields per region however large the region is, so pooling weights a small region as heavily as a large one. The area-weighted figure is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 03:31:29] R: scored in 686s
[2026-09-15 03:31:29] 
[2026-09-15 03:31:29] ===== U (N-cadherin (CDH2)) =====
[2026-09-15 03:31:29] U: paired with the H&E already carrying steps 2-9 (4XWhs5vekJ97Rx3_sflVOQ), so step 8 does not run again
[2026-09-15 03:31:29] U: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC _hwDSP734Nx1FddsedpWhA
[2026-09-15 03:31:29] U: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 03:31:30] U: step 9 found 104 invasive region(s)
[2026-09-15 03:31:30] U: step 10, registering the two slides
[2026-09-15 03:41:07] U: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 03:41:07] U: step 11, segmenting nuclei
[2026-09-15 03:43:17] U: step 11 counted 797 nuclei over 1.937 mm2 (411/mm2)
[2026-09-15 03:43:17] U: step 12 typed 797 cells, 36.6% tumour
[2026-09-15 03:43:19] U: step 13 built 1,152 cytoplasm compartments
[2026-09-15 03:43:28] U: step 14 measured 1,152 cells, mean DAB 0.5443 OD, mean stained_fraction 0.863
[2026-09-15 03:43:30] U: step 15 bins 0:24 / 1+:137 / 2+:583 / 3+:408
[2026-09-15 03:43:33] U: STEP 16 -> 100 % positive, intensity 1.5 (Moderate)   [raw 99.22 % / 0.5227 OD]
[2026-09-15 03:43:33] U:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 03:43:33] U:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 79% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 03:43:33] U:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 03:43:33] U: scored in 724s
[2026-09-15 03:43:33] 
[2026-09-15 03:43:33] ===== W (Pan-cadherin) =====
[2026-09-15 03:43:33] W: paired with the H&E already carrying steps 2-9 (4XWhs5vekJ97Rx3_sflVOQ), so step 8 does not run again
[2026-09-15 03:43:33] W: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC TiBGQDku0gwM0qjwHUldFQ
[2026-09-15 03:43:33] W: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 03:43:35] W: step 9 found 104 invasive region(s)
[2026-09-15 03:43:35] W: step 10, registering the two slides
[2026-09-15 03:52:29] W: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 03:52:29] W: step 11, segmenting nuclei
[2026-09-15 10:19:01] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-15 10:19:01]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-15 10:19:01]   scoring ['A', 'F', 'R', 'U', 'W']
[2026-09-15 10:19:01] 
[2026-09-15 10:19:01] ===== A (CD44) =====
[2026-09-15 10:19:01] A: reusing the pair already registered for this case
[2026-09-15 10:19:01] A: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC _YAC0Eaf5Hq-hlefyT362g
[2026-09-15 10:19:01] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 10:19:08] A: step 9 found 104 invasive region(s)
[2026-09-15 10:19:08] A: step 10, registering the two slides
[2026-09-15 10:19:08] A: step 11, segmenting nuclei
[2026-09-15 10:19:08] A: step 11 counted 1,064 nuclei over 1.937 mm2 (549/mm2)
[2026-09-15 10:19:08] A: step 12 typed 1,064 cells, 32.7% tumour
[2026-09-15 10:19:09] A: step 13 built 348 membrane compartments
[2026-09-15 10:19:15] A: step 14 measured 348 cells, mean DAB 0.1999 OD, mean ring_completeness 0.383
[2026-09-15 10:19:17] A: step 15 bins 0:191 / 1+:84 / 2+:60 / 3+:13
[2026-09-15 10:19:19] A: STEP 16 -> 75 % positive, intensity 1.5 (Moderate)   [raw 76.79 % / 0.4078 OD]
[2026-09-15 10:19:19] A:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 10:19:19] A:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 72% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 10:19:19] A:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 10:19:19] A:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 76.8 %; pooling every measured cell regardless of which region it came from gives 44.8 %; a plain mean of the regions gives 46.6 %. Step 11 samples a fixed number of fields per region however large the region is, so pooling weights a small region as heavily as a large one. The area-weighted figure is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 10:19:19] A: scored in 17s
[2026-09-15 10:19:19] 
[2026-09-15 10:19:19] ===== F (ABCC4 (MRP4)) =====
[2026-09-15 10:19:19] F: reusing the pair already registered for this case
[2026-09-15 10:19:19] F: H&E rLl5AdGX41FVsd8tojLhiw  IHC ZVr5wT6LHe_VyOsesxk9Sg
[2026-09-15 10:19:19] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 10:19:21] F: step 9 found 104 invasive region(s)
[2026-09-15 10:19:21] F: step 10, registering the two slides
[2026-09-15 10:19:21] F: step 11, segmenting nuclei
[2026-09-15 10:19:21] F: step 11 counted 1,244 nuclei over 1.937 mm2 (642/mm2)
[2026-09-15 10:19:21] F: step 12 typed 1,244 cells, 32.0% tumour
[2026-09-15 10:19:22] F: step 13 built 398 membrane compartments
[2026-09-15 10:19:27] F: step 14 measured 398 cells, mean DAB 0.4798 OD, mean ring_completeness 0.819
[2026-09-15 10:19:29] F: step 15 bins 0:11 / 1+:137 / 2+:175 / 3+:75
[2026-09-15 10:19:30] F: STEP 16 -> 100 % positive, intensity 1.5 (Moderate)   [raw 99.16 % / 0.5117 OD]
[2026-09-15 10:19:30] F:   caveat: OUTSIDE THE EXPECTED RANGE. ABCC4 (MRP4) has been reported between 35 % and 70 % across the cases OncoStem has read; this slide scores 100 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-15 10:19:30] F:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 10:19:30] F:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 68% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 10:19:30] F:   caveat: ALIGNMENT CONFIRMED, BY WHOM UNRECORDED. This pair was signed off before the confirmation started recording whether a person or a batch run did it. Re-confirm it on step 10 to put a person's judgement on the record.
[2026-09-15 10:19:30] F: scored in 12s
[2026-09-15 10:19:30] 
[2026-09-15 10:19:30] ===== R (ABCC11 (MRP8)) =====
[2026-09-15 10:19:30] R: reusing the pair already registered for this case
[2026-09-15 10:19:30] R: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC WIxXB152BDd6-HTQVQmLQw
[2026-09-15 10:19:30] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 10:19:31] R: step 9 found 104 invasive region(s)
[2026-09-15 10:19:31] R: step 10, registering the two slides
[2026-09-15 10:19:31] R: step 11, segmenting nuclei
[2026-09-15 10:19:31] R: step 11 counted 888 nuclei over 1.937 mm2 (458/mm2)
[2026-09-15 10:19:32] R: step 12 typed 888 cells, 31.6% tumour
[2026-09-15 10:19:32] R: step 13 built 281 membrane compartments
[2026-09-15 10:19:38] R: step 14 measured 281 cells, mean DAB 0.2683 OD, mean ring_completeness 0.656
[2026-09-15 10:19:40] R: step 15 bins 0:65 / 1+:148 / 2+:61 / 3+:7
[2026-09-15 10:19:41] R: STEP 16 -> 70 % positive, intensity 1 (Weak)   [raw 67.51 % / 0.3047 OD]
[2026-09-15 10:19:41] R:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 10:19:41] R:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 77% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 10:19:41] R:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 10:19:41] R:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 67.5 %; pooling every measured cell regardless of which region it came from gives 75.8 %; a plain mean of the regions gives 72.5 %. Step 11 samples a fixed number of fields per region however large the region is, so pooling weights a small region as heavily as a large one. The area-weighted figure is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 10:19:41] R: scored in 11s
[2026-09-15 10:19:41] 
[2026-09-15 10:19:41] ===== U (N-cadherin (CDH2)) =====
[2026-09-15 10:19:41] U: reusing the pair already registered for this case
[2026-09-15 10:19:41] U: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC _hwDSP734Nx1FddsedpWhA
[2026-09-15 10:19:41] U: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 10:19:42] U: step 9 found 104 invasive region(s)
[2026-09-15 10:19:42] U: step 10, registering the two slides
[2026-09-15 10:19:42] U: step 11, segmenting nuclei
[2026-09-15 10:19:42] U: step 11 counted 797 nuclei over 1.937 mm2 (411/mm2)
[2026-09-15 10:19:42] U: step 12 typed 797 cells, 36.6% tumour
[2026-09-15 10:19:44] U: step 13 built 292 cytoplasm compartments
[2026-09-15 10:19:50] U: step 14 measured 292 cells, mean DAB 0.4772 OD, mean stained_fraction 0.814
[2026-09-15 10:19:51] U: step 15 bins 0:10 / 1+:40 / 2+:179 / 3+:63
[2026-09-15 10:19:53] U: STEP 16 -> 100 % positive, intensity 1.5 (Moderate)   [raw 98.92 % / 0.4853 OD]
[2026-09-15 10:19:53] U:   caveat: OUTSIDE THE EXPECTED RANGE. N-cadherin (CDH2) has been reported between 70 % and 85 % across the cases OncoStem has read; this slide scores 100 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-15 10:19:53] U:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 10:19:53] U:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 79% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 10:19:53] U:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 10:19:53] U: scored in 12s
[2026-09-15 10:19:53] 
[2026-09-15 10:19:53] ===== W (Pan-cadherin) =====
[2026-09-15 10:19:53] W: reusing the pair already registered for this case
[2026-09-15 10:19:53] W: H&E 4XWhs5vekJ97Rx3_sflVOQ  IHC TiBGQDku0gwM0qjwHUldFQ
[2026-09-15 10:19:53] W: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 10:19:54] W: step 9 found 104 invasive region(s)
[2026-09-15 10:19:54] W: step 10, registering the two slides
[2026-09-15 10:19:54] W: step 11, segmenting nuclei
[2026-09-15 10:21:27] W: step 11 counted 1,160 nuclei over 1.937 mm2 (599/mm2)
[2026-09-15 10:21:27] W: step 12 typed 1,160 cells, 25.5% tumour
[2026-09-15 10:21:28] W: step 13 built 296 cytoplasm compartments
[2026-09-15 10:21:34] W: step 14 measured 296 cells, mean DAB 0.2774 OD, mean stained_fraction 0.648
[2026-09-15 10:21:36] W: step 15 bins 0:55 / 1+:120 / 2+:110 / 3+:11
[2026-09-15 10:21:38] W: STEP 16 -> 85 % positive, intensity 1 (Weak)   [raw 83.36 % / 0.2924 OD]
[2026-09-15 10:21:38] W:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 10:21:38] W:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 70% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 10:21:38] W:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 10:21:38] W: scored in 105s
[2026-09-15 10:21:38] 
[2026-09-15 10:21:38] ===== summary =====
[2026-09-15 10:21:38]   A CD44                   scored                 75% / 1.5
[2026-09-15 10:21:38]   F ABCC4 (MRP4)           scored                 100% / 1.5
[2026-09-15 10:21:38]   R ABCC11 (MRP8)          scored                 70% / 1
[2026-09-15 10:21:38]   U N-cadherin (CDH2)      scored                 100% / 1.5
[2026-09-15 10:21:38]   W Pan-cadherin           scored                 85% / 1
[2026-09-15 10:22:46] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
