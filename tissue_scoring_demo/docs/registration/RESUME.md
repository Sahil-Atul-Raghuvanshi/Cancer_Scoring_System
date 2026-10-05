# If this run stops, how to resume it

Everything below is safe to run repeatedly. Every stage is checkpointed, so re-running
skips what is done and picks up where it stopped. Nothing is recomputed just because a
restart happened.

## It does not depend on the Claude session

The stages run as detached Windows processes. An assistant session ending, a usage limit
being reached, or a terminal being closed has **no effect on them**. What they do not
survive on their own is a reboot or a hard crash, which is what the supervisor is for.

## Resume by hand

```
slide_registration\supervise.cmd
```

That is the whole answer. It loops `night.py` until every stage is done, and refuses to
start if another supervisor is already running.

## What is watching what

| piece | watches for | acts by |
|---|---|---|
| `supervise.cmd` | `night.py` exiting with stages left | re-running it (3 attempts per stage, then gives up on that stage) |
| `sentry.py` | a stage that has **hung** | killing the worker so the stage fails and the supervisor restarts it |
| `night.py` itself | a stage exceeding its wall-clock cap | `taskkill /T` on that stage's process tree |
| Startup shortcut | a reboot | starting the supervisor at logon |

`sentry.py` is the one that needed the most care, because **slow is not hung**. The
existing scoring harness learned that expensively: a log-silence watchdog killed a stage
that was working perfectly, holding one region for forty-seven minutes while burning four
cores and printing nothing. So the sentry requires *two* signals to agree before it acts -
the log silent for 45 minutes **and** the whole process tree accruing almost no CPU over
the same period - and it needs three consecutive polls to agree before it kills anything.

Restart it if it is not running:

```
tissue_scoring_demoackend\.venv\Scripts\python.exe scripts
egistration\sentry.py
```

Its own log: `data
egistration\_logs\sentry.log`. One line per poll, so you can read
back what it thought at any point in the night.

## Check progress without touching anything

```
tissue_scoring_demo\backend\.venv\Scripts\python.exe slide_registration\night.py --status
tissue_scoring_demo\backend\.venv\Scripts\python.exe slide_registration\watch.py
```

## After a reboot

A shortcut in the Startup folder restarts the supervisor at logon:

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\CancerScoringNight.lnk
```

A scheduled task would have been tidier but `schtasks` needs elevation and this session
did not have it. The Startup shortcut needs none and fires on the same event.

**Delete that shortcut when the run is finished**, or it will start a supervisor at every
logon. It will exit immediately once all stages are done, but it is still clutter.

## The stages, in order

| stage | what it does | roughly |
|---|---|---|
| `methods` | every registration method on all 30 pairs | minutes |
| `score_tiles` | score the cohort on step 9's tile regions, **skipping BEETLE** | hours |
| `score_beetle` | score the cohort on BEETLE's per-pixel boundaries | hours |
| `coverage` | tile-versus-BEETLE area tables | seconds |
| `decide` | rewrite `DECISIONS.md` | seconds |

Run one stage on its own:

```
... slide_registration\night.py --only methods
... slide_registration\night.py --from score_beetle
... slide_registration\night.py --redo methods
```

## A stage that keeps failing

The supervisor gives each stage three attempts and then treats it as finished rather than
looping forever. To give it more, clear its counter:

```
data\registration\_state\night.json     <- delete the stage's "attempts" and "state"
```

## What is written where

All of these except the plan are in `results/` at the repository root.

| file | what |
|---|---|
| `DECISIONS.md` | what worked, what did not, what is settled — **read this first** |
| `REGISTRATION-PLAN.md` (beside this file) | the full reasoning and every measurement |
| `registration_methods.csv` | one row per pair per method, with NMI and gain over outline |
| `registration_summary.csv` | the VALIS stack registration's own results |
| `oncostem_ai_scores_tiles.csv` | scores **without** BEETLE |
| `oncostem_ai_scores.csv` | scores **with** BEETLE |
| `beetle_coverage_by_{slide,region,tile}.csv` | how much of each tile region BEETLE kept |
| `data/score_archive/` (repository root) | the scores as they were before any of this started |

## Outstanding at 03:50 on 2026-09-24

**CAN_00251 needs a tiles retry.** The sentry killed it mid-run (see
`REGISTRATION-PLAN.md` §13) with only A, F and R scored. The checkpoint says `done`, so the
pass will not revisit it. After the `score_tiles` stage finishes:

```
tissue_scoring_demoackend\.venv\Scripts\python.exe score_all_slidesun_all.py --roi-source tiles --redo CAN_00251
```

Steps 2-9 are cached for that H&E, so this should only cost markers U and W.

Check whether any other case came up short:

```
tissue_scoring_demoackend\.venv\Scripts\python.exe -c "import json,glob;[print(f, len(json.load(open(f))['markers'])) for f in glob.glob('score_all_slides/results_tiles/*.json')]"
```
