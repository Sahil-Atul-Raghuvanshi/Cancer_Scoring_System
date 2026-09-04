@echo off
REM ===========================================================================
REM  Cancer Scoring System - set up the web app
REM
REM  A double-clickable wrapper around setup.py. Every argument is passed
REM  straight through, so these all work:
REM
REM      setup.bat
REM      setup.bat --check
REM      setup.bat --drive-folder https://drive.google.com/drive/folders/XXXX
REM      setup.bat --with-training
REM ===========================================================================

setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python was not found on PATH.
    echo          Install Python 3.11 or newer, tick "Add python.exe to PATH",
    echo          and run this again.
    echo.
    pause
    exit /b 1
)

python setup.py %*
set "SETUP_EXIT=%errorlevel%"

if not "%SETUP_EXIT%"=="0" (
    echo.
    echo  Setup reported problems. Read the Summary above - each line names
    echo  the file or dependency that is missing and where it comes from.
    echo.
)

REM Only pause when double-clicked, so the script stays usable from a shell.
echo %CMDCMDLINE% | find /i "/c" >nul
if not errorlevel 1 pause

endlocal
exit /b %SETUP_EXIT%
