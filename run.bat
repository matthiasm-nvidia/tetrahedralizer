@echo off
setlocal
cd /d "%~dp0"

set "EXE=build\Release\tetrahedralizer.exe"
if not exist "%EXE%" (
    echo ERROR: %EXE% does not exist. Run build.bat first.
    exit /b 1
)

"%EXE%" %*
endlocal
