"""The measurement half of the pipeline: steps 14, 15 and 16.

Three packages sit above this one - `step16_per_cell_measurement`,
`step17_intensity_binning`, `step18_aggregate` - and each owns its own
arithmetic. What lives here is the one thing all three read and none of them
owns: **the cut points**, five independent sets, one per antibody, in a
versioned file rather than inline in code.

Why here and not in step 15, where the cuts are applied? Because step 14 needs
the positivity cut too - a membrane bin counts as stained when its DAB optical
density clears it - and step 16 needs the band table. A constant that three
steps read is not any one of their private business, and putting it in step 15
would have made steps 14 and 16 import across a sibling boundary to get at it.
"""
