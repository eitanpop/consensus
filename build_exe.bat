@echo off
rem Builds a standalone Windows executable in dist\Consensus\
rem One-time: pip install pyinstaller
cd /d "%~dp0"
python -m PyInstaller --noconfirm --windowed --name "Consensus" ^
  --add-data "prompts;prompts" ^
  consensus_app.py
if errorlevel 1 (
  echo Build failed. Is PyInstaller installed?  pip install pyinstaller
  pause
  exit /b 1
)
echo.
echo Built: dist\Consensus\Consensus.exe
echo The exe still uses the claude, codex and gemini CLIs installed on this PC.
pause
