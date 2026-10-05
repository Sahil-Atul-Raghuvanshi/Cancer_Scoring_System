# Scoring every case, unattended

Runs the full pipeline over every case in `data/original/oncostem_slides`, writes one CSV
holding every marker's percent positive and intensity, and files each finished case into
`v<N>_data/data/history` (the active data version - see `data_versions.py`).

## The one file with all the scores

```
v<N>_data/results/oncostem_ai_scores.csv  <- the active version's results/
```

One row per case per marker, thirty rows when the pass is complete. `percent` and
`intensity` are the deliverable — the same two numbers OncoStem's readers give — and
everything after them is the working that produced them. It is **rebuilt from scratch**
after every marker, so it is always consistent: a crash cannot leave a half-row in it,
and a case run twice cannot appear twice.

## Running it

```cmd
score_all_slides\run_overnight.cmd                  every case not already done
score_all_slides\run_overnight.cmd --deadline 9.5   start no new case after 9.5 hours
score_all_slides\run_overnight.cmd --redo CAN_00270 run one again regardless
```

Detached, so it outlives the shell that starts it. **Everything is restartable**: run it
again tomorrow and it continues from the checkpoint rather than repeating.

## Watching it

```cmd
tissue_scoring_demo\backend\.venv\Scripts\python.exe score_all_slides\watch.py
tissue_scoring_demo\backend\.venv\Scripts\python.exe score_all_slides\watch.py --follow
```

Read-only and safe while the pass is running. Prints each case's state, how long its log
has been silent, and the last three lines of the case in progress.

## What is where

| Path | What it holds |
| --- | --- |
| `v<N>_data/results/oncostem_ai_scores.csv` | **every score, all cases, one file** |
| `v<N>_data/data/score_all_slides/results/<CASE>.json` | one case's markers, the CSV's only source |
| `v<N>_data/data/score_all_slides/state/checkpoint.json` | which cases are done; what a restart reads |
| `v<N>_data/data/score_all_slides/logs/run_all.log` | the orchestrator: one line per case |
| `v<N>_data/data/score_all_slides/logs/<CASE>.log` | one case, stage by stage — where to look when one fails |
| `v<N>_data/data/score_all_slides/logs/console.log` | anything the wrapper itself printed |
| `v<N>_data/data/history/<CASE>/` | each finished case's work, **moved** here, not copied |

## The pieces

- **`run_all.py`** — the orchestrator. Spawns one case at a time and watches it. A case
  that dies is marked `crashed` and retried on the next run; a case whose log goes silent
  for 90 minutes is treated as hung, killed, and marked `stalled`. That is what turns one
  bad case from "the night was wasted" into "one case was lost and the other five ran".
- **`run_case.py`** — one case in its own process. Five markers, two attempts each, the
  second restarting the alignment. Writes the per-case record after **every** marker, so a
  crash in marker four cannot lose the three that scored before it.
- **`pipeline.py`** — paths, the checkpoint, the CSV. Both JSON and CSV are written
  through an atomic replace, so a power cut cannot truncate them.
- **`watch.py`** — the read-only status view.

## Two things to know about the numbers

**Nobody ticked the regions.** Step 10 is a screen where a person chooses which invasive
regions go to BEETLE. A batch has nobody, so the run takes `roi_selection_service`'s own
default — the coverage rule the pipeline would have applied for itself — and the record
stores `chosenByPerson: false` beside it.

**Nobody looked at the alignment.** Step 12's gate exists so a person can confirm the two
slides landed on the same tissue. The runner confirms it with a machine stamp, and step 18
turns that stamp into a caveat printed beside every number. Those caveats travel into the
CSV's `caveats` column — they are not decoration, and a number carrying one has not been
checked by a human at the point the gate exists to have a human check.

`cuts_provisional` is `true` on every row until the marker cut points are fitted. The
percentages are measurements; the intensity *bands* they fall into are not yet calibrated.
