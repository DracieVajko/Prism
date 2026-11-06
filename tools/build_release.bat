@echo off
cd /d D:\AI\DiffusionServer\android
set "JAVA_HOME=D:\AI\tools\jdk17"
set "PATH=D:\AI\tools\jdk17\bin;%PATH%"
gradlew.bat :app:assembleRelease --no-daemon > D:\AI\tools\build_release.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\AI\tools\build_release.log
