@echo off
chcp 65001 > nul

:: 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ========================================
    echo   ⚠️  需要管理员权限
    echo ========================================
    echo.
    echo 请右键点击此文件，选择"以管理员身份运行"
    echo.
    pause
    exit /b 1
)

echo ========================================
echo   配置Windows防火墙 - 允许端口访问
echo ========================================
echo.
echo 正在添加防火墙规则...
echo.

:: 添加后端端口规则
netsh advfirewall firewall add rule name="Ref4D Django Backend 8000" dir=in action=allow protocol=TCP localport=8000 >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ 后端端口 8000 已允许
) else (
    echo ℹ 后端端口 8000 规则可能已存在
)

:: 添加前端端口规则
netsh advfirewall firewall add rule name="Ref4D Vue Frontend 8080" dir=in action=allow protocol=TCP localport=8080 >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ 前端端口 8080 已允许
) else (
    echo ℹ 前端端口 8080 规则可能已存在
)

echo.
echo ========================================
echo   ✅ 防火墙配置完成！
echo ========================================
echo.
echo 现在其他设备可以访问你的服务器了！
echo.
echo 下一步：
echo   1. 双击运行 "启动后端.bat"
echo   2. 双击运行 "启动前端.bat"
echo   3. 使用显示的IP地址访问
echo.
echo 如需删除防火墙规则，运行：
echo   netsh advfirewall firewall delete rule name="Ref4D Django Backend 8000"
echo   netsh advfirewall firewall delete rule name="Ref4D Vue Frontend 8080"
echo.
pause
