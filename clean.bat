@echo off
echo ===============================================
echo  VideoMaxx - Clean All Data
echo ===============================================
echo.
echo This will DELETE:
echo   - All uploaded videos (data\raw_videos\)
echo   - All labels (data\labels.csv)
echo   - All job outputs (outputs\)
echo   - All trained model artifacts (ml_artifacts\)
echo.
set /p confirm="Are you sure? Type YES to confirm: "
if /i not "%confirm%"=="YES" (
    echo Cancelled.
    pause
    exit /b 0
)

cd /d "%~dp0"

echo Cleaning raw videos...
if exist "data\raw_videos" rd /s /q "data\raw_videos"
mkdir "data\raw_videos"

echo Cleaning labels...
if exist "data\labels.csv" del "data\labels.csv"

echo Cleaning outputs...
if exist "outputs" rd /s /q "outputs"
mkdir "outputs"

echo Cleaning model artifacts...
if exist "ml_artifacts" rd /s /q "ml_artifacts"
mkdir "ml_artifacts"

echo.
echo Done. All data cleared.
pause
