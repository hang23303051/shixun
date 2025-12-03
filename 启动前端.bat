@echo off
chcp 65001 > nul
echo ========================================
echo   启动 Ref4D 前端服务器 (局域网访问)
echo ========================================
echo.

cd /d "%~dp0frontend"

echo [1/2] 获取本机IP地址...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP:~1%
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
