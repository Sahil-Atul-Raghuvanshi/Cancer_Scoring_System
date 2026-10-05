@echo off
REM Keep the unattended night running until every stage is done.
REM
REM **This is what makes the run survive things nobody is awake to notice.** The stages
REM themselves are detached OS processes and already outlive the session that started
REM them - an assistant session ending, a terminal closing, a usage limit being reached
REM have no effect on them whatsoever. What they do NOT survive is the machine rebooting,
REM or a stage crashing hard enough to take its own process down.
REM
REM So this loop re-runs `night.py` until it reports nothing left to do. That is safe to
REM do any number of times because every stage is checkpointed: a completed stage is
REM skipped, a half-finished case is resumed from its own per-case record, and nothing is
REM ever recomputed just because the supervisor restarted.
REM
REM Register it to run at logon as well, so a reboot resumes the night:
REM
REM   schtasks /Create /TN CancerScoringNight /TR "<this file>" /SC ONLOGON /RL HIGHEST /F
REM
REM Remove it when the run is finished:
REM
REM   schtasks /Delete /TN CancerScoringNight /F

setlocal

set ROOT=C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System
set PY=%ROOT%\tissue_scoring_demo\backend\.venv\Scripts\python.exe
set HERE=%ROOT%\slide_registration
REM The data version, asked of data_versions.py so it follows the same rule as the app:
REM CSS_DATA_VERSION, then storage\ACTIVE_DATA_VERSION, falling back to the latest
REM version this code can open.
set VER=
for /f %%v in ('"%PY%" "%ROOT%\data_versions.py" --active') do set VER=%%v
if "%VER%"=="" set VER=v1
set LOGDIR=%ROOT%\storage\%VER%_data\data\registration\_logs

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

cd /d %ROOT%\tissue_scoring_demo\backend

title Cancer scoring - unattended night - DO NOT CLOSE

REM A guard against two supervisors running at once, which would have two copies of a
REM stage writing the same files. Whoever holds the lock file wins; the other exits.
if exist "%LOGDIR%\supervisor.lock" (
  echo A supervisor is already running. Delete supervisor.lock to override.
  echo %DATE% %TIME% second supervisor refused >> "%LOGDIR%\supervisor.log"
  exit /b 0
)
echo %DATE% %TIME% >"%LOGDIR%\supervisor.lock"

:loop
echo ============================================================
echo  %DATE% %TIME%  starting/resuming the night
echo ============================================================
echo %DATE% %TIME% attempt >> "%LOGDIR%\supervisor.log"

"%PY%" -u "%HERE%\night.py" >> "%LOGDIR%\night_console.log" 2>&1

REM night.py exits 0 once every stage is either done or has failed permanently. Ask it
REM whether anything is still outstanding rather than inferring from the exit code.
"%PY%" "%HERE%\night_pending.py"
if %ERRORLEVEL% EQU 0 goto finished

echo %DATE% %TIME% stages remain, retrying in 120s >> "%LOGDIR%\supervisor.log"
REM timeout rather than ping: this window may have no console when run from a scheduled
REM task, and /nobreak keeps a stray keypress from skipping the wait.
timeout /t 120 /nobreak >nul
goto loop

:finished
echo %DATE% %TIME% all stages complete >> "%LOGDIR%\supervisor.log"
del "%LOGDIR%\supervisor.lock" >nul 2>&1
echo.
echo ============================================================
echo  FINISHED - read storage\%VER%_data\results\DECISIONS.md
echo ============================================================
endlocal
