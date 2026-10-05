[2026-09-15 17:43:54] case CAN_00270 at C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\data\original\oncostem_slides\CAN_00270
[2026-09-15 17:43:54]   found ['A', 'F', 'HE', 'R', 'U', 'W'], missing []
[2026-09-15 17:43:54]   scoring ['A', 'F']
[2026-09-15 17:43:54] 
[2026-09-15 17:43:54] ===== A (CD44) =====
[2026-09-15 17:43:54] A: reusing the pair already registered for this case
[2026-09-15 17:43:54] A: H&E uC8eGezagEjBHNSoxjUF1Q  IHC alEtP1bYCEZkbuqS3oekRA
[2026-09-15 17:43:54] A: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 17:43:59] A: step 9 found 104 invasive region(s)
[2026-09-15 17:43:59] A: step 10, registering the two slides
[2026-09-15 17:46:42] A: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 17:46:42] A: step 11, segmenting nuclei
[2026-09-15 17:50:33] A: step 11 counted 2,435 nuclei over 5.166 mm2 (471/mm2)
[2026-09-15 17:50:33] A: step 12 typed 2,435 cells, 33.0% tumour
[2026-09-15 17:50:38] A: step 13 built 803 membrane compartments
[2026-09-15 17:50:42] A: step 14 measured 803 cells, mean DAB 0.2483 OD, mean ring_completeness 0.474
[2026-09-15 17:50:46] A: step 15 bins 0:369 / 1+:186 / 2+:214 / 3+:34
[2026-09-15 17:50:50] A: STEP 16 -> 65 % positive, intensity 1 (Weak)   [raw 64.15 % / 0.3871 OD]
[2026-09-15 17:50:50] A:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 17:50:50] A:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 76% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 17:50:50] A:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 17:50:50] A:   caveat: REGION WEIGHTING MATTERS HERE. Combined by area the answer is 64.2 %; pooling every measured cell regardless of which region it came from gives 53.5 %; a plain mean of the regions gives 27.6 %. Step 11 now allocates its fields in proportion to region area, so those first two figures are close where the tumour is one dominant mass and drift apart as the guaranteed minimum field per region lifts small regions above their share. The plain mean is the outlier by design: it gives a 0.25 mm2 fragment the same say as a 22 mm2 mass. The area-weighted figure is reported, which is the reading OncoStem's own procedure implies - the entire slide scanned, every field averaged (SOP 4.2) - though whether they weight by area or by cell count is still unanswered (Q3).
[2026-09-15 17:50:50] A: scored in 416s
[2026-09-15 17:50:50] 
[2026-09-15 17:50:50] ===== F (ABCC4 (MRP4)) =====
[2026-09-15 17:50:50] F: reusing the pair already registered for this case
[2026-09-15 17:50:50] F: H&E uC8eGezagEjBHNSoxjUF1Q  IHC pSgsl4NGvz55UMwoJRwxhw
[2026-09-15 17:50:50] F: steps 2-9 on the H&E (step 8 is the slow one)
[2026-09-15 17:50:50] F: step 9 found 104 invasive region(s)
[2026-09-15 17:50:50] F: step 10, registering the two slides
[2026-09-15 17:53:15] F: step 10 confirmed BY MACHINE - no person has looked at the panels, and every score from this pair carries that caveat
[2026-09-15 17:53:15] F: step 11, segmenting nuclei
[2026-09-15 17:57:13] F: step 11 counted 3,890 nuclei over 5.166 mm2 (753/mm2)
[2026-09-15 17:57:13] F: step 12 typed 3,890 cells, 31.9% tumour
[2026-09-15 17:57:19] F: step 13 built 1,242 membrane compartments
[2026-09-15 17:57:24] F: step 14 measured 1,242 cells, mean DAB 0.4712 OD, mean ring_completeness 0.821
[2026-09-15 17:57:28] F: step 15 bins 0:43 / 1+:402 / 2+:586 / 3+:211
[2026-09-15 17:57:33] F: STEP 16 -> 95 % positive, intensity 1.5 (Moderate)   [raw 96.00 % / 0.4969 OD]
[2026-09-15 17:57:33] F:   caveat: OUTSIDE THE EXPECTED RANGE. ABCC4 (MRP4) has been reported between 35 % and 70 % across the cases OncoStem has read; this slide scores 95 %. That is a flag for human review, not an error - but with cut points that have not been fitted, a result this far out is more likely to be the cut points than the tissue.
[2026-09-15 17:57:33] F:   caveat: PROVISIONAL CUT POINTS. They have not been fitted against the 120 pathologist readings, because that sheet is not on disk. The percentage and the intensity both move with these numbers, so treat the pair as a demonstration that the pipeline computes the contract, not as a measurement to act on.
[2026-09-15 17:57:33] F:   caveat: DENOMINATOR INCOMPLETE. This slide yielded 62% fewer nuclei per mm2 than the case's own H&E inside the same regions. Serial sections of one block hold the same cells, so that gap is a segmentation failure rather than biology - under heavy DAB the counterstain is too weak for nuclear boundaries to survive deconvolution. Every nucleus missed is a cell out of the denominator, and missed cells are disproportionately the strongly stained ones, so this inflates the percentage.
[2026-09-15 17:57:33] F:   caveat: ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 programmatically so it could proceed unattended. No person has looked at the two panels.
[2026-09-15 17:57:33] F: scored in 403s
[2026-09-15 17:57:33] 
[2026-09-15 17:57:33] ===== summary =====
[2026-09-15 17:57:33]   A CD44                   scored                 65% / 1
[2026-09-15 17:57:33]   F ABCC4 (MRP4)           scored                 95% / 1.5
[2026-09-15 17:59:21] workbook: C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System\CAN_00270_scores.xlsx
