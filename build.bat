@echo off

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b

cmake --build build
if errorlevel 1 exit /b
