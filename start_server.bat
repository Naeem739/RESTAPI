@echo off
echo Starting Paddy Disease Detection Flask API...
echo.
echo Checking if Python is installed...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Checking if required packages are installed...
pip install -r requirements.txt

echo.
echo Starting Flask server...
echo The API will be available at: http://127.0.0.1:5000
echo Health check: http://127.0.0.1:5000/api/health
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
