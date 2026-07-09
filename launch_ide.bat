@echo off
chcp 65001 >nul
title One-Code IDE
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║     One-Code IDE Server                ║
echo  ║     DeepSeek V4 Pro + L4 Self-Evolution ║
echo  ╚══════════════════════════════════════════╝
echo.

REM Check if port is free
set PORT=8765
netstat -ano | findstr ":%PORT% " >nul 2>&1
if %errorlevel% equ 0 (
    echo  [!] Port %PORT% is in use. Trying to open browser only...
    start http://localhost:%PORT%
    goto :end
)

REM Start IDE server
start "One-Code IDE" /MIN python ide_server.py

REM Wait for server to start
echo  等待服务启动...
timeout /t 3 /nobreak >nul

REM Open browser
start http://localhost:%PORT%

echo.
echo   IDE 运行在 http://localhost:%PORT%
echo   关闭此窗口不会停止 IDE 服务
echo.

:end
pause
