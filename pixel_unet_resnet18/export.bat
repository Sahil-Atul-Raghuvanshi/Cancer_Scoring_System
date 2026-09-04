@echo off
REM ===========================================================================
REM  pixel_unet_resnet18 - cut the (image, mask) tile pairs, in its own window
REM
REM  ~20 minutes, ~660 MB. SAFE TO RUN AGAIN: every region writes a shard when
REM  its tiles are on disk, and a region whose shard matches the current
REM  settings is skipped. An interrupted export resumes; a finished one is a
REM  no-op that takes a second.
REM
REM    export.bat                  all 270 regions
REM    export.bat --bcss-only      skip the BEETLE-labelled BRACS regions
REM    export.bat --limit 6        a smoke run
REM ===========================================================================

setlocal
cd /d "%~dp0"

set "VENV=..\Breast_Cancer_IHC_Tissue_Scoring_Demo\backend\.venv\Scripts\python.exe"
if not exist "%VENV%" (
    echo  [ERROR] Interpreter not found at %VENV%
    pause
    exit /b 1
)

echo.
echo  ==========================================================
echo   pixel_unet_resnet18 - cutting tile pairs
echo   reading  ..\bcss_bracs_hchannel_resnet18\data  (read-only)
echo   writing  data\seg_tiles
echo  ==========================================================
echo.

start "pixel-unet export" cmd /k ""%~dp0%VENV%" -u "%~dp0scripts\01_export_seg_tiles.py" %*"

echo  Export started in a new window titled "pixel-unet export".
echo.
endlocal
exit /b 0
