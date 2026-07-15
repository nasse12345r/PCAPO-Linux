@echo off
echo =======================================================
echo   FYP DDoS Detection System Dashboard
echo =======================================================

:: Add Wireshark to PATH for this session only so tshark can be found
set PATH=%PATH%;D:\Wireshark

:: Check if Python is installed
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    pause
    goto :EOF
)

:: Fix OpenBLAS memory issue for Numpy/Scikit-Learn
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1

:: Start the browser with a 5 second delay to allow Flask and ML models to load
echo Starting browser...
start "" cmd /c "timeout /t 5 >nul && start http://localhost:5000"

:: Start the Flask app
echo Starting Flask Server...
py app.py

pause
