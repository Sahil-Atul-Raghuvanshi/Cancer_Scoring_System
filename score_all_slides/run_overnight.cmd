@echo off
REM Score every case in the slide library, unattended, overnight.
REM
REM A .cmd wrapper rather than a direct call for two reasons. The agent shell that
REM writes these files cannot spawn a nested powershell.exe (it fails with EPERM),
REM and a job measured in hours has to outlive the session that started it.
REM
REM Everything below is restartable. run_all.py reads its checkpoint before it
REM decides anything, so killing this and running it again tomorrow continues the
REM pass rather than repeating it.
REM
REM   run_overnight.cmd                 every case not already done
REM   run_overnight.cmd --deadline 9.5  start no new case after 9.5 hours
REM   run_overnight.cmd --redo CAN_00270

setlocal

set ROOT=C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System
set PY=%ROOT%\tissue_scoring_demo\backend\.venv\Scripts\python.exe
set HERE=%ROOT%\score_all_slides

REM The data version, asked of data_versions.py so it follows the same rule as the app:
REM CSS_DATA_VERSION, then storage\ACTIVE_DATA_VERSION, falling back to the latest
REM version this code can open. Run output lives under
REM %ROOT%\storage\vN_data\, not under the shared storage\data\ folder.
set VER=
for /f %%v in ('"%PY%" "%ROOT%\data_versions.py" --active') do set VER=%%v
if "%VER%"=="" set VER=v1
set VDATA=%ROOT%\storage\%VER%_data
if not exist "%VDATA%\data\score_all_slides\logs" mkdir "%VDATA%\data\score_all_slides\logs"

if not exist "%HERE%\logs" mkdir "%HERE%\logs"

REM cwd is the backend, matching how every other script in this repository is run,
REM so anything resolving a path relative to the working directory finds it.
cd /d %ROOT%\tissue_scoring_demo\backend

title Scoring every case - DO NOT CLOSE THIS WINDOW

echo ============================================================
echo    Scoring every case. Leave this window open.
echo    Closing it stops the run.
echo.
echo    Also written to:
echo      storage\%VER%_data\data\score_all_slides\logs\run_all.log   one line per case
echo      storage\%VER%_data\data\score_all_slides\logs\CASE.log      stage by stage
echo      storage\%VER%_data\results\oncostem_ai_scores.csv      the scores
echo ============================================================
echo.

REM stdout is left on the console so this window shows what it is doing. An empty
REM window on an overnight job is one a person reasonably assumes has died, and
REM closing it would stop the run - so the blank console was a hazard, not a saving.
REM Nothing is lost by not redirecting: every line `pipeline.say` prints is written
REM to run_all.log as well.
REM
REM stderr still goes to a file, because a traceback is the one thing worth having
REM after the window has gone.
REM
REM -u so the console updates as the run goes rather than when a buffer fills.
"%PY%" -u "%HERE%\run_all.py" %* 2>> "%VDATA%\data\score_all_slides\logs\console.log"

echo.
echo ============================================================
echo    FINISHED - exit code %ERRORLEVEL%
echo    Scores: storage\%VER%_data\results\oncostem_ai_scores.csv
echo ============================================================
echo EXIT CODE %ERRORLEVEL% >> "%VDATA%\data\score_all_slides\logs\console.log"
REM Holds the window open so the final summary is readable in the morning rather
REM than vanishing with the process.
pause
endlocal
