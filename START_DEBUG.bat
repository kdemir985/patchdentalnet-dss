@echo off
REM Launches the DSS desktop app WITH a visible console window,
REM for troubleshooting (errors/logs are printed here). The normal desktop
REM shortcut ("PatchDentalNet DSS") uses pythonw.exe instead, which hides
REM this console.
cd /d "%~dp0backend"
"..\..\kds\backend\.venv\Scripts\python.exe" app_window.py
pause
