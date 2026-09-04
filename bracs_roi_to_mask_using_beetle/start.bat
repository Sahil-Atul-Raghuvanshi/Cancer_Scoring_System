@echo off
REM ===========================================================================
REM  BRACS DCIS ROI to mask - start the app
REM
REM  One service, not two. The frontend is three static files served by FastAPI
REM  itself, so there is no Node toolchain and nothing to wait for twice.
REM
REM  The interpreter is the demo backend's virtual environment, reused rather
REM  than duplicated: it already has torch 2.8 (CPU), FastAPI, uvicorn, Pillow,
REM  scipy and numpy, which is everything this app needs. A second venv would
REM  mean a second 200 MB torch download to run the same code.
REM ===========================================================================

setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:8100"
set "VENV=..\Breast_Cancer_IHC_Tissue_Scoring_Demo\backend\.venv\Scripts\python.exe"

echo.
echo  ==========================================================
echo   BRACS DCIS - ROI to mask
echo   BEETLE teacher  -^>  BCSS-coded mask  -^>  approach 1 tiles
echo  ==========================================================
echo.

REM --------------------------------------------------------------------------
REM  1/3  The interpreter
REM --------------------------------------------------------------------------
if not exist "%VENV%" (
    echo  [ERROR] The demo backend's virtual environment was not found at:
    echo          %VENV%
    echo.
    echo          This app reuses it rather than creating its own. Run the demo's
    echo          own start.bat once to create it, then retry.
    goto :fail
)
echo  [1/3] Interpreter found.

REM --------------------------------------------------------------------------
REM  2/3  The inputs. Checked here rather than at first click, because a missing
REM       1.9 GB of weights should not surface as a stack trace four screens in.
REM --------------------------------------------------------------------------
set "MISSING="
if not exist "data\dcis\train" set "MISSING=%MISSING% BRACS-regions(run resume_dcis.sh)"
if not exist "..\beetle_teacher_resnet18\data\model\model.zip" set "MISSING=%MISSING% BEETLE-weights(Zenodo-16812932)"
if not exist "..\bcss_bracs_hchannel_resnet18\src\export.py" set "MISSING=%MISSING% approach-1-src"

if not "%MISSING%"=="" (
    echo  [ERROR] Missing inputs:%MISSING%
    goto :fail
)
echo  [2/3] BRACS regions, BEETLE weights and approach 1's exporter are all present.

REM --------------------------------------------------------------------------
REM  3/3  The server. No --reload: it doubles the process tree for no gain on
REM       something whose startup cost is a 1.9 GB zip index, and stop.bat then
REM       has one process to kill instead of two.
REM --------------------------------------------------------------------------
echo  [3/3] Starting on %APP_URL% ...
start "BRACS Mask App" cmd /k "cd /d "%~dp0backend" && "%~dp0%VENV%" -m uvicorn bracs_app.main:app --host 127.0.0.1 --port 8100"

echo        waiting for the API to answer...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "for($i=0;$i -lt 90;$i++){ try { $r = Invoke-WebRequest -Uri '%APP_URL%/api/health' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
    echo        [WARN] No answer within 90s. Check the 'BRACS Mask App' window.
    goto :done
)
echo        up.

echo.
echo  Opening %APP_URL% ...
start "" "%APP_URL%"

:done
echo.
echo  ==========================================================
echo   App        %APP_URL%
echo   API docs   %APP_URL%/docs
echo.
echo   Outputs are written to data\runs\^<region^>\ :
echo     *_bcss_codes.png   the mask, in BCSS's own label codes
echo     overlay.png        what the model saw, over the tissue
echo     tiles\             224px haematoxylin tiles, by class
echo     manifest.json      provenance and the verdict
echo.
echo   Run stop.bat to shut it down.
echo  ==========================================================
echo.
endlocal
exit /b 0

:fail
echo.
echo  [FAILED] Startup aborted. See the error above.
echo.
endlocal
pause
exit /b 1
