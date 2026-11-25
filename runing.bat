
@echo off
chcp 65001 >nul
setlocal

title CECI Coastline Analysis System

REM ==========================================
REM 1. Python installation
REM ==========================================

set PYTHON_VERSION=3.10.12
set PYTHON_INSTALLER=%TEMP%\python-%PYTHON_VERSION%-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

echo.
echo ==========================================
echo Checking Python...
echo ==========================================

python --version >nul 2>&1

IF ERRORLEVEL 1 (
    echo Python not found!
    echo Downloading Python %PYTHON_VERSION%...

    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"

    IF NOT EXIST "%PYTHON_INSTALLER%" (
        echo Python download failed!
        pause
        exit /b 1
    )

    echo Installing Python...

    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

    REM Refresh PATH
    set "PATH=%PATH%;C:\Program Files\Python310;C:\Program Files\Python310\Scripts"

    python --version

    IF ERRORLEVEL 1 (
        echo Python installation failed!
        pause
        exit /b 1
    )
) ELSE (
    echo Python already installed:
    python --version
)

REM ==========================================
REM 2. Google Earth Pro installation
REM ==========================================

echo.
echo ==========================================
echo Checking Google Earth Pro...
echo ==========================================

set GE_INSTALLED=0

IF EXIST "%ProgramFiles%\Google\Google Earth Pro\client\googleearth.exe" (
    set GE_INSTALLED=1
)

IF EXIST "%ProgramFiles(x86)%\Google\Google Earth Pro\client\googleearth.exe" (
    set GE_INSTALLED=1
)

IF "%GE_INSTALLED%"=="0" (

    echo Google Earth Pro not found!
    echo Downloading Google Earth Pro...

    set GE_INSTALLER=%TEMP%\GoogleEarthProSetup.exe

    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://dl.google.com/dl/earth/client/advanced/current/GoogleEarthProWin.exe' -OutFile '%TEMP%\GoogleEarthProSetup.exe'"

    IF NOT EXIST "%TEMP%\GoogleEarthProSetup.exe" (
        echo Google Earth Pro download failed!
        pause
        exit /b 1
    )

    echo Installing Google Earth Pro...

    REM Run installer
    "%TEMP%\GoogleEarthProSetup.exe"

    echo Google Earth Pro installation started.
    echo Please wait for installation to finish.

    timeout /t 10

) ELSE (
    echo Google Earth Pro already installed.
)

REM ==========================================
REM 3. Create Python virtual environment
REM ==========================================

echo.
echo ==========================================
echo Checking virtual environment...
echo ==========================================

set VENV_NAME=env

IF NOT EXIST "%VENV_NAME%\Scripts\activate.bat" (

    echo Creating virtual environment...

    python -m venv %VENV_NAME%

    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment!
        pause
        exit /b 1
    )

    call "%VENV_NAME%\Scripts\activate.bat"

    REM Upgrade pip
    python -m pip install --upgrade pip

    REM Install dependencies
    IF EXIST requirements.txt (
        echo Installing dependencies...
        python -m pip install -r requirements.txt
    ) ELSE (
        echo requirements.txt not found!
    )

) ELSE (

    echo Virtual environment already exists.
    call "%VENV_NAME%\Scripts\activate.bat"

)

REM ==========================================
REM 4. Run Streamlit application
REM ==========================================

echo.
echo ==========================================
echo Environment setup complete!
echo ==========================================

call "%VENV_NAME%\Scripts\activate.bat"

echo Starting CECI application...

streamlit run CECI_app.py

pause

