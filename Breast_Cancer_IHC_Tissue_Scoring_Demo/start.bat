@echo off
REM ===========================================================================
REM  Breast Cancer IHC Tissue Scoring - start the demo
REM
REM  Starts the FastAPI backend first, waits until it actually answers, then
REM  starts the Vite frontend. Each service gets its own terminal window so
REM  its logs stay readable. Run stop.bat to shut both down.
REM ===========================================================================

setlocal
cd /d "%~dp0"

set "BACKEND_URL=http://127.0.0.1:8000"
set "FRONTEND_URL=http://localhost:5173"

echo.
echo  ==========================================================
echo   Breast Cancer IHC Tissue Scoring
echo   starting backend, then frontend
echo  ==========================================================
echo.

REM --------------------------------------------------------------------------
REM  Prerequisites
REM --------------------------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python was not found on PATH. Install Python 3.11+ and retry.
    goto :fail
)

where node >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js was not found on PATH. Install Node 18+ and retry.
    goto :fail
)

REM --------------------------------------------------------------------------
REM  1/4  Backend dependencies
REM --------------------------------------------------------------------------
if exist "backend\.venv\Scripts\python.exe" (
    echo  [1/4] Backend virtual environment found.
) else (
    echo  [1/4] Creating backend virtual environment ^(first run, this takes a minute^)...
    python -m venv "backend\.venv"
    if errorlevel 1 goto :fail
    "backend\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "backend\.venv\Scripts\python.exe" -m pip install -r "backend\requirements.txt" --quiet
    if errorlevel 1 goto :fail
    echo        dependencies installed.
)

REM --------------------------------------------------------------------------
REM  2/4  Backend - started first, because the frontend proxies /api to it
REM --------------------------------------------------------------------------
echo  [2/4] Starting backend on %BACKEND_URL% ...
start "IHC Backend" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo        waiting for the API to answer...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "for($i=0;$i -lt 60;$i++){ try { $r = Invoke-WebRequest -Uri '%BACKEND_URL%/api/v1/health' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
    echo        [WARN] The API did not answer within 60s. Check the 'IHC Backend' window.
) else (
    echo        backend is up.
)

REM --------------------------------------------------------------------------
REM  3/4  Frontend dependencies
REM --------------------------------------------------------------------------
if exist "frontend\node_modules" (
    echo  [3/4] Frontend dependencies found.
) else (
    echo  [3/4] Installing frontend dependencies ^(first run, this takes a minute^)...
    pushd "frontend"
    call npm install
    set "NPM_FAILED=%errorlevel%"
    popd
    if not "%NPM_FAILED%"=="0" goto :fail
    echo        dependencies installed.
)

REM --------------------------------------------------------------------------
REM  4/4  Frontend
REM --------------------------------------------------------------------------
echo  [4/4] Starting frontend on %FRONTEND_URL% ...
start "IHC Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo        waiting for the dev server...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "for($i=0;$i -lt 60;$i++){ try { $r = Invoke-WebRequest -Uri '%FRONTEND_URL%/' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
    echo        [WARN] The dev server did not answer within 60s. Check the 'IHC Frontend' window.
    goto :done
)
echo        frontend is up.

echo.
echo  Opening %FRONTEND_URL% in your browser...
start "" "%FRONTEND_URL%"

:done
echo.
echo  ==========================================================
echo   App        %FRONTEND_URL%
echo   API        %BACKEND_URL%
echo   API docs   %BACKEND_URL%/docs
echo.
echo   Two terminal windows are now open, one per service.
echo   Run stop.bat to shut both down.
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
