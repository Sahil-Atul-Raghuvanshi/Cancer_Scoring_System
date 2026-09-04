@echo off
REM ===========================================================================
REM  Breast Cancer IHC Tissue Scoring - stop the demo
REM
REM  Shuts down in reverse order: frontend first, then backend.
REM
REM  Killing "the process on the port" is not enough:
REM
REM    * `uvicorn --reload` runs a reloader parent plus a worker child, and
REM      `npm run dev` spawns vite as a child. Killing one leaves the other.
REM    * A child can inherit the listening socket, so the port can stay held by
REM      a process id that no longer exists.
REM    * On Windows a surviving worker and a freshly started server can BOTH
REM      bind the same port - start.bat then reports success while requests are
REM      answered by stale code.
REM
REM  So: kill every listener, every descendant, anything whose command line
REM  still names the service, then verify the port is actually free.
REM ===========================================================================

setlocal
cd /d "%~dp0"

echo.
echo  ==========================================================
echo   Breast Cancer IHC Tissue Scoring
echo   stopping frontend, then backend
echo  ==========================================================
echo.

echo  [1/3] Stopping frontend ^(port 5173^)...
call :killport 5173 "vite"

echo  [2/3] Stopping backend ^(port 8000^)...
call :killport 8000 "uvicorn app.main:app"

echo  [3/3] Closing service terminal windows...
taskkill /f /fi "WINDOWTITLE eq IHC Frontend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq IHC Backend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Administrator:  IHC Frontend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Administrator:  IHC Backend*" >nul 2>&1

echo.
echo  ==========================================================
echo   Both services stopped.
echo  ==========================================================
echo.
endlocal
exit /b 0

REM --------------------------------------------------------------------------
REM  :killport <port> <command-line-fragment>
REM --------------------------------------------------------------------------
:killport
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-service.ps1" -Port %~1 -Match %2
exit /b 0
