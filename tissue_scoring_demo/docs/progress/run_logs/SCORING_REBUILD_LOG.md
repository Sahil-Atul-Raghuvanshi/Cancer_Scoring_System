[2026-09-15 15:14:18] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-15 15:14:18]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-15 15:14:18]   scoring ['A', 'F', 'R', 'U', 'W']
[2026-09-15 15:14:18] 
[2026-09-15 15:14:18] ===== A (CD44) =====
[2026-09-15 15:14:18] A: reusing the pair already registered for this case
[2026-09-15 15:14:18] A: H&E uC8eGezagEjBHNSoxjUF1Q  IHC alEtP1bYCEZkbuqS3oekRA
[2026-09-15 15:14:18] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 15:22:13] A: step 9 found 95 invasive region(s)
[2026-09-15 15:22:13] A: step 10, registering the two slides
[2026-09-15 15:27:45] A: step 10 refused - no residual error was recorded, so how far apart the matched features ended up is unknown and cannot be checked (VALIS recorded no registration summary. Its output directory may be too deep for the filesystem to write into - check that <dst_dir>/<name>/data/ exists.)
[2026-09-15 15:27:45] A: alignment_refused in 807s
[2026-09-15 15:27:45] 
[2026-09-15 15:27:45] ===== F (ABCC4 (MRP4)) =====
[2026-09-15 15:27:45] F: paired with the H&E already carrying steps 2-9 (uC8eGezagEjBHNSoxjUF1Q), so step 8 does not run again
[2026-09-15 15:27:45] F: H&E uC8eGezagEjBHNSoxjUF1Q  IHC pSgsl4NGvz55UMwoJRwxhw
[2026-09-15 15:27:45] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 15:27:45] F: step 9 found 95 invasive region(s)
[2026-09-15 15:27:45] F: step 10, registering the two slides
[2026-09-15 15:33:17] F: step 10 refused - no residual error was recorded, so how far apart the matched features ended up is unknown and cannot be checked (VALIS recorded no registration summary. Its output directory may be too deep for the filesystem to write into - check that <dst_dir>/<name>/data/ exists.)
[2026-09-15 15:33:17] F: alignment_refused in 333s
[2026-09-15 15:33:17] 
[2026-09-15 15:33:17] ===== R (ABCC11 (MRP8)) =====
[2026-09-15 15:33:17] R: paired with the H&E already carrying steps 2-9 (uC8eGezagEjBHNSoxjUF1Q), so step 8 does not run again
[2026-09-15 15:33:17] R: H&E uC8eGezagEjBHNSoxjUF1Q  IHC 4hKfO50nC0MZ4L85eqKWhw
[2026-09-15 15:33:17] R: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 15:33:18] R: step 9 found 95 invasive region(s)
[2026-09-15 15:33:18] R: step 10, registering the two slides
[2026-09-15 15:43:40] R: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 15:43:40] R: step 11, segmenting nuclei
[2026-09-15 15:49:39] R: step 11 counted 1,515 nuclei over 5.166 mm2 (293/mm2)
[2026-09-15 15:49:40] R: step 12 typed 1,515 cells, 37.8% tumour
[2026-09-15 15:49:50] R: step 13 built 572 membrane compartments
[2026-09-15 15:50:01] R: step 14 measured 572 cells, mean DAB 0.2528 OD, mean ring_completeness 0.563
[2026-09-15 15:50:11] R: step 15 bins 0:174 / 1+:268 / 2+:113 / 3+:17
[2026-09-15 15:50:21] R: STEP 16 -> 65 % positive, intensity 1 (Weak)   [raw 64.65 % / 0.3004 OD]
[2026-09-15 15:50:21] R:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 15:50:21] R:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 77% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 15:50:21] R:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 15:50:21] R: scored in 1024s
[2026-09-15 15:50:21] 
[2026-09-15 15:50:21] ===== U (N-cadherin (CDH2)) =====
[2026-09-15 15:50:22] U: paired with the H&E already carrying steps 2-9 (uC8eGezagEjBHNSoxjUF1Q), so step 8 does not run again
[2026-09-15 15:50:22] U: H&E uC8eGezagEjBHNSoxjUF1Q  IHC cclFCbmFPTgjS0U1LCOvaA
[2026-09-15 15:50:22] U: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 16:03:36] U: step 9 found 95 invasive region(s)
[2026-09-15 16:03:36] U: step 10, registering the two slides
[2026-09-15 16:13:58] U: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 16:13:58] U: step 11, segmenting nuclei
[2026-09-15 16:18:49] U: step 11 counted 2,478 nuclei over 5.166 mm2 (480/mm2)
[2026-09-15 16:18:49] U: step 12 typed 2,478 cells, 36.1% tumour
[2026-09-15 16:18:55] U: step 13 built 894 cytoplasm compartments
[2026-09-15 16:19:00] U: step 14 measured 894 cells, mean DAB 0.4293 OD, mean stained_fraction 0.807
[2026-09-15 16:19:05] U: step 15 bins 0:18 / 1+:221 / 2+:501 / 3+:154
[2026-09-15 16:19:10] U: STEP 16 -> 100 % positive, intensity 1.5 (Moderate)   [raw 97.90 % / 0.4399 OD]
[2026-09-15 16:19:10] U:   caveat: OUTSIDE THE EXPECTED RANGE. N-cadherin (CDH2) has been reported between 70 % and 85 % across the cases OncoStem has read; this slide scores 100 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-15 16:19:10] U:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 16:19:10] U:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 62% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 16:19:10] U:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 16:19:10] U: scored in 1729s
[2026-09-15 16:19:10] 
[2026-09-15 16:19:10] ===== W (Pan-cadherin) =====
[2026-09-15 16:19:10] W: paired with the H&E already carrying steps 2-9 (uC8eGezagEjBHNSoxjUF1Q), so step 8 does not run again
[2026-09-15 16:19:10] W: H&E uC8eGezagEjBHNSoxjUF1Q  IHC TwKXqXLe-Pt-KxHc2aqXCw
[2026-09-15 16:19:10] W: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 16:25:09] W: step 9 found 104 invasive region(s)
[2026-09-15 16:25:09] W: step 10, registering the two slides
[2026-09-15 16:32:51] W: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 16:32:51] W: step 11, segmenting nuclei
[2026-09-15 16:36:29] W: step 11 counted 2,497 nuclei over 5.166 mm2 (483/mm2)
[2026-09-15 16:36:29] W: step 12 typed 2,497 cells, 25.7% tumour
[2026-09-15 16:36:33] W: step 13 built 642 cytoplasm compartments
[2026-09-15 16:36:37] W: step 14 measured 642 cells, mean DAB 0.2877 OD, mean stained_fraction 0.655
[2026-09-15 16:36:41] W: step 15 bins 0:100 / 1+:287 / 2+:217 / 3+:38
[2026-09-15 16:36:45] W: STEP 16 -> 85 % positive, intensity 1 (Weak)   [raw 84.41 % / 0.3137 OD]
[2026-09-15 16:36:45] W:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 16:36:45] W:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 16:36:45] W:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 16:36:45] W:   caveat: Combining the invasive regions by area and averaging them plainly differ by 7.5 points. Area-weighted is reported; which OncoStem uses is unanswered (Q3).
[2026-09-15 16:36:45] W: scored in 1055s
[2026-09-15 16:36:45] 
[2026-09-15 16:36:45] ===== summary =====
[2026-09-15 16:36:45]   A CD44                   alignment_refused      -
[2026-09-15 16:36:45]   F ABCC4 (MRP4)           alignment_refused      -
[2026-09-15 16:36:45]   R ABCC11 (MRP8)          scored                 65% / 1
[2026-09-15 16:36:45]   U N-cadherin (CDH2)      scored                 100% / 1.5
[2026-09-15 16:36:45]   W Pan-cadherin           scored                 85% / 1
[2026-09-15 16:37:34] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
