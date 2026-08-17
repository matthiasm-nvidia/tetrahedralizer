@echo off
setlocal

cd /d "%~dp0"
set "BUILD_DIR=build"
set "CONFIG=Release"

rem Locate Visual Studio and its bundled Ninja.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VSINSTALL=%%i"
    )
)

if not defined VSINSTALL (
    echo ERROR: Visual Studio with C++ tools not found.
    exit /b 1
)

set "VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA_DIR=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if not exist "%VCVARS%" (
    echo ERROR: vcvars64.bat not found at %VCVARS%
    exit /b 1
)
if not exist "%NINJA_DIR%\ninja.exe" (
    echo ERROR: ninja.exe not found at %NINJA_DIR%
    exit /b 1
)

set "PATH=%NINJA_DIR%;%PATH%"
call "%VCVARS%"

if not exist "%BUILD_DIR%\build.ninja" (
    echo Configuring...
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_CUDA_ARCHITECTURES="86-real;120-real;90-virtual"
    if errorlevel 1 exit /b 1
)

echo Building tetrahedralizer_tests ^(%CONFIG%^)...
cmake --build "%BUILD_DIR%" --config %CONFIG% --target tetrahedralizer_tests
if errorlevel 1 exit /b 1

set "TEST_EXE=%BUILD_DIR%\%CONFIG%\tetrahedralizer_tests.exe"
if not exist "%TEST_EXE%" (
    echo ERROR: test executable not found at %TEST_EXE%
    exit /b 1
)

echo.
echo Running %TEST_EXE%
"%TEST_EXE%"
set "TEST_STATUS=%ERRORLEVEL%"
echo.
if not "%TEST_STATUS%"=="0" (
    echo TESTS FAILED
    exit /b %TEST_STATUS%
)

echo TESTS PASSED
endlocal
