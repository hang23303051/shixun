@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
echo ========================================
echo   启动 Ref4D 前端服务器 (局域网访问)
echo ========================================
echo.

cd /d "%~dp0frontend"

echo [1/2] 获取本机IP地址...
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

echo [2/2] 启动Vue开发服务器...
echo.
echo ========================================
echo   🌐 前端访问地址：
echo ========================================
echo   📍 本机访问:    http://localhost:8080
echo   📍 局域网访问:   http://%IP%:8080
echo.
echo   📱 手机/平板访问: http://%IP%:8080
echo   💻 其他电脑访问:  http://%IP%:8080
echo.
echo   ⚠️ 请确保后端服务器已启动！
echo   按 Ctrl+C 停止服务器
echo ========================================
echo.

npm run serve

pause
