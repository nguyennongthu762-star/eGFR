@echo off
rem Mo backend + frontend trong 2 cua so rieng, doi backend nap xong mo hinh roi mo trinh duyet.
rem Dong 2 cua so do de tat.
cd /d "%~dp0"
start "CKD backend (cong 8000)" cmd /k "cd backend && .venv\Scripts\python -m uvicorn app.main:app --port 8000"
start "CKD frontend (cong 5173)" cmd /k "cd frontend && npm run dev"
echo Dang doi backend nap mo hinh (co the mat 10-60 giay)...
set /a tries=0
:wait
set /a tries+=1
curl -s -o nul http://localhost:8000/api/health && goto ready
if %tries% geq 90 goto ready
ping -n 2 127.0.0.1 >nul
goto wait
:ready
start "" http://localhost:5173
