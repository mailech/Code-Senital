@echo off
echo Starting Self-Healing Codebase Sentinel...
echo.

echo Testing application components...
python quick_test.py
echo.

echo Starting FastAPI server...
echo Open http://localhost:8000/dashboard in your browser
echo Press Ctrl+C to stop the server
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
