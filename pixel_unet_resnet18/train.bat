@echo off
REM ===========================================================================
REM  pixel_unet_resnet18 - train the segmentation model, in its own window
REM
REM  Opens a titled console and streams progress there: a line every few
REM  batches with the loss, the rate, and how long the epoch and the whole run
REM  have left. The window stays open when the run ends (`cmd /k`) so a result
REM  reached at 3 a.m. is still on screen in the morning.
REM
REM  SAFE TO RUN AGAIN. A checkpoint lands after every epoch, and re-running
REM  resumes from it - so closing the window, a reboot, or Ctrl-C costs at most
REM  the epoch in flight. Running it after the epochs are done just re-scores
REM  and rewrites the report.
REM
REM    train.bat                     6000 tiles/epoch, 12 epochs, ~9 h
REM    train.bat --smoke             2 epochs on 200 tiles, a few minutes
REM    train.bat --tiles-per-epoch 4000
REM ===========================================================================

setlocal
cd /d "%~dp0"

set "VENV=..\Breast_Cancer_IHC_Tissue_Scoring_Demo\backend\.venv\Scripts\python.exe"

if not exist "%VENV%" (
    echo  [ERROR] The interpreter was not found at:
    echo          %VENV%
    echo.
    echo          This pipeline reuses the demo backend's virtual environment.
    echo          Run the demo's own start.bat once to create it, then retry.
    pause
    exit /b 1
)

if not exist "data\seg_tiles\manifest.json" (
    echo  [ERROR] No tile manifest at data\seg_tiles\manifest.json
    echo.
    echo          Cut the tiles first ^(about 20 minutes, and re-runnable^):
    echo            export.bat
    pause
    exit /b 1
)

REM  Default to the 6000-tiles-per-epoch recipe, but let any argument through so
REM  `train.bat --smoke` and `train.bat --epochs 20` both work. %* is empty on a
REM  bare double-click, which is the common case.
set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--tiles-per-epoch 12569 --batch-size 16"

echo.
echo  ==========================================================
echo   pixel_unet_resnet18 - segmentation training
echo   %ARGS%
echo.
echo   Progress prints every few batches. Ctrl-C is safe: the
echo   last completed epoch is already checkpointed, and running
echo   this again resumes from it.
echo  ==========================================================
echo.

REM  -u is not optional. Without it Python buffers stdout in 8 KB blocks, and a
REM  progress line written every 30 seconds would appear in bursts minutes apart -
REM  which looks exactly like a hung process.
start "pixel-unet training" cmd /k ""%~dp0%VENV%" -u "%~dp0scripts\02_train_seg.py" %ARGS%"

echo  Training started in a new window titled "pixel-unet training".
echo.
endlocal
exit /b 0
