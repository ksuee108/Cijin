@echo off
REM -------------------------------
REM Check Python, auto-install Python 3.10 if not exists
REM -------------------------------

REM Set Python version and download path
set PYTHON_VERSION=3.10.12
set PYTHON_INSTALLER=%TEMP%\python-%PYTHON_VERSION%-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

REM Check if Python exists
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python not found in the system!
    echo Will automatically download and install Python %PYTHON_VERSION%...
    
    REM Download Python installer
    powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"

    REM Silent install Python, add to PATH
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

    REM Verify installation success
    python --version >nul 2>&1
    IF ERRORLEVEL 1 (
        echo Python installation failed, please install manually!
        pause
        exit /b
    ) ELSE (
        echo Python installation complete: 
        python --version
    )
) ELSE (
    echo Found Python:
    python --version
)
set VENV_NAME=env
IF NOT EXIST "%VENV_NAME%\Scripts\activate.bat" (
    echo Virtual environment "%VENV_NAME%" does not exist, creating automatically...
    
    REM Create virtual environment
    python -m venv %VENV_NAME%
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment, please confirm Python is installed.
        pause
        exit /b
    )

    REM Activate virtual environment
    call %VENV_NAME%\Scripts\activate.bat

    REM Upgrade pip
    python -m pip install --upgrade pip

    REM Install dependencies
    if exist requirements.txt (
        pip install -r requirements.txt
    ) ELSE (
        echo requirements.txt not found, please install dependencies manually.
    )
) ELSE (
    REM If already exists, activate virtual environment directly
    call %VENV_NAME%\Scripts\activate.bat
)

echo ✅ Virtual environment setup complete

REM Activate virtual environment
call env\Scripts\activate.bat

REM Run application
streamlit run CECI_app.py

REM Keep command window after execution
pause
