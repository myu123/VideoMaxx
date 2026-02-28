@echo off
echo ===============================================
echo  VideoMaxx - Starting Frontend (Angular)
echo ===============================================

cd /d "%~dp0\frontend"

if not exist "node_modules" (
    echo Installing npm packages...
    npm install
)

echo Starting Angular dev server on http://localhost:4200
npx ng serve --open
