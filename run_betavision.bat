@echo off
rem BetaVision launcher: double-click to start the web app in its own window.
rem
rem The Python packages live in one shared place, %LOCALAPPDATA%\BetaVision\venv, so a
rem new copy of BetaVision (a fresh download, another folder) starts straight away and
rem only installs anything when requirements.txt has changed. The first run needs
rem internet and takes a few minutes; it also adds BetaVision to the Start menu and desktop.
rem
rem   run_betavision.bat            set up if needed, then start
rem   run_betavision.bat --setup    set up (or update) only, don't start
title BetaVision
cd /d "%~dp0"
setlocal

set "HOME_DIR=%LOCALAPPDATA%\BetaVision"
set "VENV=%HOME_DIR%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\requirements.installed.txt"

if not exist "%PY%" (
    echo Setting up BetaVision for the first time. This needs internet and takes a few minutes...
    echo.
    if not exist "%HOME_DIR%" mkdir "%HOME_DIR%"
    py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
    if not exist "%PY%" goto nopython
    set "FIRST=1"
)

rem install only when requirements.txt differs from what's installed
if exist "%STAMP%" fc /b "requirements.txt" "%STAMP%" >nul 2>&1 && goto installed
echo Installing BetaVision's Python packages...
"%PY%" -m pip install --disable-pip-version-check -q --upgrade pip
"%PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
copy /y "requirements.txt" "%STAMP%" >nul
echo.
:installed

call :shortcuts

if /i "%~1"=="--setup" (
    echo BetaVision is set up. Open it from the Start menu, or double-click run_betavision.bat.
    exit /b 0
)

"%PY%" -B -m betavision_app --window
if errorlevel 1 (
    echo.
    echo BetaVision stopped with an error. See the messages above.
    pause
)
exit /b

rem ---------------------------------------------------------------- Start menu + desktop shortcuts
:shortcuts
rem The Start menu entry always points at this copy (so moving or re-downloading BetaVision just works).
rem The desktop shortcut is only made the first time; delete it if you don't want it.
set "BV_ICON=%~dp0betavision_app\static\icons\betavision.ico"
set "BV_TARGET=%~f0"
set "BV_DIR=%~dp0"
set "BV_FIRST=%FIRST%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = New-Object -ComObject WScript.Shell;" ^
  "$links = @([IO.Path]::Combine([Environment]::GetFolderPath('Programs'), 'BetaVision.lnk'));" ^
  "if ($env:BV_FIRST) { $links += [IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'BetaVision.lnk') };" ^
  "foreach ($p in $links) { $l = $s.CreateShortcut($p); $l.TargetPath = $env:BV_TARGET; $l.WorkingDirectory = $env:BV_DIR;" ^
  "  $l.IconLocation = $env:BV_ICON; $l.WindowStyle = 7; $l.Description = 'BetaVision climbing video analyzer'; $l.Save() }" >nul 2>&1
exit /b 0

:nopython
echo Couldn't find Python. Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then double-click this file again.
pause
exit /b 1

:failed
echo.
echo Installing the requirements failed. Check your internet connection and double-click again.
del "%STAMP%" 2>nul
pause
exit /b 1
