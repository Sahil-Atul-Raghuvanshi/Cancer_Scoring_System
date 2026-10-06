@echo off
REM Keep the P-03 night running until night.py reports it finished.
REM
REM The stages are checkpointed, so re-running night.py after a crash resumes rather than
REM repeats. This loop is what survives a stage crashing hard enough to take night.py with
REM it. A reboot is not survived; re-run this file (or launch.py) to resume.

setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\tissue_scoring_demo\backend\.venv\Scripts\python.exe"
set "HERE=%ROOT%\p03_nuclei"
for /f %%v in ('"%PY%" "%ROOT%\data_versions.py" --active') do set VER=%%v
if "%VER%"=="" set VER=v1
set "OUT=%ROOT%\storage\%VER%_data\data\p03_nuclei"
if not exist "%OUT%\logs" mkdir "%OUT%\logs"
cd /d "%HERE%"

if exist "%OUT%\supervisor.lock" (
  echo %DATE% %TIME% second supervisor refused >> "%OUT%\logs\supervisor.log"
  exit /b 0
)
echo %DATE% %TIME% > "%OUT%\supervisor.lock"

set /a LOOPS=0
:loop
set /a LOOPS+=1
echo %DATE% %TIME% run %LOOPS% >> "%OUT%\logs\supervisor.log"
"%PY%" -u night.py >> "%OUT%\logs\night_console.log" 2>&1
"%PY%" -c "import common,sys; sys.exit(0 if (common.read_json(common.STATE/'night.json',{}) or {}).get('finished') else 1)"
if %ERRORLEVEL%==0 goto done
if %LOOPS% GEQ 20 goto done
REM `timeout` fails without a console; a hidden supervisor has none.
ping -n 61 127.0.0.1 > nul
goto loop

:done
echo %DATE% %TIME% supervisor done after %LOOPS% run(s) >> "%OUT%\logs\supervisor.log"
del "%OUT%\supervisor.lock"
endlocal
