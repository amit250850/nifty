@echo off
set SRC=%USERPROFILE%\Documents\NiftySignalBot
set DST=D:\Nifty\NiftySignalBot

echo Source: %SRC%
echo Dest:   %DST%
echo.

if not exist "%SRC%" (
    echo ERROR: Source folder not found: %SRC%
    echo Check your Documents folder name and try again.
    pause
    exit /b 1
)

if not exist "%DST%" (
    echo ERROR: Destination not found: %DST%
    pause
    exit /b 1
)

copy "%SRC%\main.py"                        "%DST%\main.py" /Y
copy "%SRC%\modules\option_chain.py"        "%DST%\modules\option_chain.py" /Y
copy "%SRC%\modules\strike_selector.py"     "%DST%\modules\strike_selector.py" /Y
copy "%SRC%\modules\position_guard.py"      "%DST%\modules\position_guard.py" /Y

echo.
echo Done - files copied to %DST%
pause
