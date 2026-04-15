@echo off
:: ------------------------------------------------------------------
:: HKN PoliTO · Associate Portrait Pipeline — zero-setup launcher
:: ------------------------------------------------------------------
:: 1. Creates a Python virtual environment (.\venv) if missing.
:: 2. Activates it.
:: 3. Upgrades pip + installs requirements.txt (only when needed).
:: 4. Launches the Streamlit UI.
::
:: Requires Python 3.10-3.13 installed via python.org (includes the
:: Python Launcher "py.exe"). Python 3.11 or 3.12 recommended.
:: ------------------------------------------------------------------

setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo  HKN PoliTO · Associate Portrait Pipeline
echo  ------------------------------------------------
echo.

:: ---- Find best available Python via the Windows Launcher ----------
set PYBIN=
for %%v in (3.12 3.11 3.13 3.10) do (
    if not defined PYBIN (
        py -%%v --version >nul 2>&1
        if !errorlevel! == 0 (
            set PYBIN=py -%%v
            for /f "tokens=2" %%V in ('py -%%v --version 2^>^&1') do (
                echo  · using py -%%v  ^(Python %%V^)
            )
        )
    )
)

:: Fallback: bare "python" (e.g. conda, winget installs without Launcher)
if not defined PYBIN (
    python --version >nul 2>&1
    if !errorlevel! == 0 (
        set PYBIN=python
        for /f "tokens=2" %%V in ('python --version 2^>^&1') do (
            echo  · using python  ^(Python %%V^)
        )
    )
)

if not defined PYBIN (
    echo.
    echo  ERROR: no Python 3 interpreter found.
    echo  Install Python 3.12 from https://python.org and retry.
    echo  Make sure "Add Python to PATH" is checked during install.
    echo.
    pause
    exit /b 1
)

:: ---- Create or reuse venv -----------------------------------------
if not exist "venv\" (
    echo  · venv not found · creating .\venv
    %PYBIN% -m venv venv
    if !errorlevel! neq 0 (
        echo  ERROR: failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    :: Check for Python version mismatch between existing venv and selected interpreter
    for /f %%V in ('venv\Scripts\python.exe -c "import sys; print(\"%%d.%%d\" %% sys.version_info[:2])" 2^>nul') do set VENV_V=%%V
    for /f %%V in ('%PYBIN% -c "import sys; print(\"%%d.%%d\" %% sys.version_info[:2])"') do set CUR_V=%%V

    if not "!VENV_V!" == "!CUR_V!" (
        echo  ! existing venv uses Python !VENV_V!, selected is !CUR_V!
        echo    rebuilding .\venv to match
        rmdir /s /q venv
        %PYBIN% -m venv venv
        if !errorlevel! neq 0 (
            echo  ERROR: failed to recreate virtual environment.
            pause
            exit /b 1
        )
    ) else (
        echo  · venv present · reusing ^(Python !VENV_V!^)
    )
)

call venv\Scripts\activate.bat

:: ---- Install dependencies (SHA-256 stamp to skip if unchanged) ----
echo  · upgrading pip · installing requirements
python -m pip install --upgrade pip wheel setuptools --quiet

set STAMP=venv\.requirements.sha256

:: certutil computes SHA256 on Windows; extract just the hash line
for /f "skip=1 tokens=*" %%H in ('certutil -hashfile requirements.txt SHA256 2^>nul') do (
    if not "%%H" == "CertUtil: -hashfile command completed successfully." (
        set CURRENT_HASH=%%H
        goto :hash_done
    )
)
:hash_done

set SAVED_HASH=
if exist "%STAMP%" (
    set /p SAVED_HASH=<"%STAMP%"
)

if "!CURRENT_HASH!" == "!SAVED_HASH!" (
    echo  · requirements unchanged · skipping install
) else (
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo  ERROR: pip install failed. Check the output above.
        pause
        exit /b 1
    )
    echo !CURRENT_HASH!>"%STAMP%"
)

:: ---- Launch -------------------------------------------------------
echo.
echo  · launching Streamlit · Ctrl-C to quit
echo.
streamlit run app.py --server.headless true --browser.gatherUsageStats false

endlocal
