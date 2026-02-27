@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  STT-TTS Installer (requires system CUDA)
echo ============================================
echo.

set "ROOT=%~dp0"
cd /d "%ROOT%"

:: Check for existing micromamba
if not exist "%ROOT%micromamba.exe" (
    echo [1/5] Downloading micromamba...
    powershell -Command "Invoke-WebRequest -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile 'micromamba.tar.bz2'"
    if errorlevel 1 (
        echo ERROR: Failed to download micromamba
        pause
        exit /b 1
    )
    
    echo Extracting micromamba...
    tar -xf micromamba.tar.bz2
    if exist "Library\bin\micromamba.exe" (
        move "Library\bin\micromamba.exe" "%ROOT%micromamba.exe" >nul
    )
    
    :: Cleanup extracted folders
    rmdir /s /q Library 2>nul
    del micromamba.tar.bz2 2>nul
    
    if not exist "%ROOT%micromamba.exe" (
        echo ERROR: Failed to extract micromamba
        pause
        exit /b 1
    )
) else (
    echo [1/5] micromamba.exe already exists, skipping download.
)

:: Create environment with CUDA toolkit
echo.
echo [2/5] Creating Python environment with CUDA support...
if exist "%ROOT%.env" (
    echo Environment already exists. Removing old environment...
    rmdir /s /q "%ROOT%.env"
)

:: Note: cudnn is omitted - assumes system cuDNN is installed
:: Add cudnn=9.* to the line below if you need it bundled
"%ROOT%micromamba.exe" create -y -p "%ROOT%.env" ^
    python=3.11 ^
    pip ^
    -c conda-forge
if errorlevel 1 (
    echo ERROR: Failed to create environment
    pause
    exit /b 1
)

:: Install PyTorch with CUDA 12.4
echo.
echo [3/5] Installing PyTorch with CUDA 12.4...
"%ROOT%micromamba.exe" run -p "%ROOT%.env" python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

:: Install core dependencies
echo.
echo [4/5] Installing dependencies...
"%ROOT%micromamba.exe" run -p "%ROOT%.env" python -m pip install ^
    faster-whisper ^
    silero-vad ^
    kokoro-onnx ^
    onnxruntime-gpu ^
    sounddevice ^
    soundfile ^
    numpy ^
    scipy ^
    pyaudio ^
    keyboard ^
    praat-parselmouth

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

:: Create run.bat
echo.
echo [5/5] Creating run.bat...
(
echo @echo off
echo setlocal
echo set "ROOT=%%~dp0"
echo cd /d "%%ROOT%%"
echo "%%ROOT%%micromamba.exe" run -p "%%ROOT%%.env" python -m app.main %%*
echo if errorlevel 1 pause
) > "%ROOT%run.bat"

:: Create models directory
if not exist "%ROOT%models" mkdir "%ROOT%models"

echo.
echo ============================================
echo  Installation complete!
echo ============================================
echo.
echo Run 'run.bat' to start the application.
echo On first run, you will be asked to select audio devices.
echo.
pause
