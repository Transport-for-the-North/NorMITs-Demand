@echo off

set anaconda_path=%USERPROFILE%\Anaconda3
set miniconda_path=%USERPROFILE%\Miniconda3
set anaconda_activate=%anaconda_path%\Scripts\activate.bat
set miniconda_activate=%miniconda_path%\Scripts\activate.bat
set env_name=NorMITs-Demand
set env_file=requirements.txt

REM Check if conda is already in PATH
WHERE conda
IF errorlevel 0 (
    goto create_env
)

REM Activate conda from default Miniconda or Anaconda folder if it isn't in PATH
IF EXIST %anaconda_activate% (
    set env_path=%anaconda_path%\envs\%env_name%
    set activate=%anaconda_activate%
) ELSE (
    IF EXIST %miniconda_activate% (
        set env_path=%miniconda_path%\envs\%env_name%
        set activate=%miniconda_activate%
    ) ELSE (
        echo Please ensure Anaconda3 is installed to %USERPROFILE%\Anaconda3
        echo or Miniconda3 is installed to %USERPROFILE%\Miniconda3.
        echo Anaconda: https://www.anaconda.com/products/individual
        echo Miniconda: https://docs.conda.io/en/latest/miniconda.html
        echo If the filepath is different, please add conda folder to PATH.
        goto end
    )
)

call %activate%

:create_env
REM Create environment, overwriting if one already exists
echo Creating %env_name%
call conda env list | find /i "%env_name%"
IF not errorlevel 1 (
    echo Found existing copy of %env_name%, removing
    call conda env remove -n %env_name%
)

call conda create -n %env_name% --file %env_file% -c conda-forge -y
echo %env_name% created, you may exit this installer.

:end
pause
