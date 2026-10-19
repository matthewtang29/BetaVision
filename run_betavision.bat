@echo off
rem BetaVision launcher: double-click to start the web app and open it in your browser.
rem First run creates a .venv folder here and installs requirements.txt (needs internet, a few minutes).
title BetaVision
cd /d "%~dp0"

if exist ".venv\.installed" goto run

echo Setting up BetaVision for the first time. This needs internet and takes a few minutes...
echo.
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" goto nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo ok> ".venv\.installed"
echo.

:run
call ".venv\Scripts\activate.bat"
python -B -m betavision_app --open
if errorlevel 1 (
    echo.
    echo BetaVision stopped with an error. See the messages above.
    pause
)
exit /b

:nopython
echo Couldn't find Python. Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then double-click this file again.
pause
exit /b 1

:failed
echo.
echo Installing the requirements failed. Check your internet connection and double-click again.
pause
exit /b 1
