@echo off
chcp 65001 > nul
echo ========================================
echo   启动 Ref4D 后端服务器 (局域网访问)
echo ========================================
echo.

cd /d "%~dp0backend"

echo [1/3] 获取本机IP地址...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP:~1%
echo ✓ 本机IP: %IP%
echo.

echo [2/3] 激活虚拟环境...
call venv\Scripts\activate.bat
echo.

echo [3/3] 启动Django服务器...
echo.
echo ========================================
echo   🚀 后端服务器访问地址：
echo ========================================
echo   📍 本机访问:   http://localhost:8000
echo   📍 局域网访问:  http://%IP%:8000
echo   📍 API接口:    http://%IP%:8000/api/
echo.
echo   按 Ctrl+C 停止服务器
echo ========================================
echo.

python manage.py runserver 0.0.0.0:8000

pause
