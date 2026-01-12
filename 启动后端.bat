@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
echo ========================================
echo   启动 Ref4D 后端服务器 (局域网访问)
echo ========================================
echo.

cd /d "%~dp0backend"

echo [1/3] 获取本机IP地址...
:: 优先获取192.168或172.16网段的IP（真实局域网IP）
:: 跳过虚拟网卡（如VPN、虚拟机等）
set IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1"') do (
    set TEMP_IP=%%a
    set TEMP_IP=!TEMP_IP:~1!
    
    :: 优先级1: 192.168.x.x
    echo !TEMP_IP! | findstr /r "^192\.168\." >nul && (
        set IP=!TEMP_IP!
        goto :found
    )
    
    :: 优先级2: 172.16-31.x.x
    echo !TEMP_IP! | findstr /r "^172\.1[6-9]\." >nul && (
        set IP=!TEMP_IP!
        goto :found
    )
    echo !TEMP_IP! | findstr /r "^172\.2[0-9]\." >nul && (
        set IP=!TEMP_IP!
        goto :found
    )
    echo !TEMP_IP! | findstr /r "^172\.3[0-1]\." >nul && (
        set IP=!TEMP_IP!
        goto :found
    )
    
    :: 优先级3: 10.x.x.x
    echo !TEMP_IP! | findstr /r "^10\." >nul && (
        set IP=!TEMP_IP!
        goto :found
    )
)

:: 如果没找到标准局域网IP，使用第一个非127.0.0.1的IP
if "%IP%"=="" (
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1"') do (
        set IP=%%a
        set IP=!IP:~1!
        goto :found
    )
)

:found
echo ✓ 本机IP: %IP%
echo.

echo [2/3] 激活虚拟环境...
call ..\venv\Scripts\activate.bat
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
