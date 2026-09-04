@echo off
REM ===========================================================================
REM  BRACS DCIS ROI to mask - stop the app
REM
REM  Killing whatever owns port 8100 is not sufficient on Windows, for the
REM  reasons the demo's own stop.bat sets out at length: a child can inherit the
REM  listening socket, so the port stays held by a process id that no longer
REM  exists, and two processes can end up bound to the same port so the next
REM  start looks healthy while stale code answers.
REM
REM  This app makes that worse in one specific way. A segmentation run holds a
REM  370 MB checkpoint and several region-sized float32 accumulators in a worker
REM  thread; killing the listener without its tree leaves that memory allocated
REM  until the orphan is noticed by hand.
REM
REM  So the demo's stop-service.ps1 is reused - one script, tested, rather than
REM  a second copy of the same logic that will drift from it.
REM ===========================================================================

setlocal
cd /d "%~dp0"

set "HELPER=..\Breast_Cancer_IHC_Tissue_Scoring_Demo\scripts\stop-service.ps1"

echo.
echo  ==========================================================
echo   BRACS DCIS - ROI to mask
echo   stopping
echo  ==========================================================
echo.

if not exist "%HELPER%" (
    echo  [WARN] %HELPER% not found; falling back to a plain port kill.
    echo         An orphaned worker thread may survive this.
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8100 .*LISTENING"') do (
        echo        stopping pid %%p
        taskkill /f /pid %%p >nul 2>&1
    )
    goto :windows
)

echo  [1/2] Stopping the server ^(port 8100^)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER%" -Port 8100 -Match "bracs_app.main:app"

:windows
echo  [2/2] Closing the service terminal window...
taskkill /f /fi "WINDOWTITLE eq BRACS Mask App*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Administrator:  BRACS Mask App*" >nul 2>&1

echo.
echo  ==========================================================
echo   Stopped. Anything already written to data\runs\ is kept.
echo  ==========================================================
echo.
endlocal
exit /b 0
