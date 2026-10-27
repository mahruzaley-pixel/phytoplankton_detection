@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   GIT REPOSITORY SETUP FOR GITHUB PUSH
echo ============================================================

:: Cek keberadaan git di PATH atau Program Files
where git >nul 2>nul
if %errorlevel% equ 0 (
    set "GIT_CMD=git"
) else (
    if exist "C:\Program Files\Git\cmd\git.exe" (
        set "GIT_CMD=C:\Program Files\Git\cmd\git.exe"
    ) else if exist "C:\Program Files\Git\bin\git.exe" (
        set "GIT_CMD=C:\Program Files\Git\bin\git.exe"
    ) else (
        echo [ERROR] Git tidak ditemukan di komputer. Silakan install Git for Windows terlebih dahulu.
        pause
        exit /b 1
    )
)

echo [+] Menggunakan Git: "!GIT_CMD!"
"!GIT_CMD!" --version

echo.
echo [+] Inisialisasi Git repository...
"!GIT_CMD!" init

:: Cek apakah identitas Git sudah diset
"!GIT_CMD!" config user.name >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [!] Identitas Git belum diset di komputer ini.
    set /p GIT_USER="Masukkan Nama Anda untuk Git (contoh: John Doe): "
    set /p GIT_EMAIL="Masukkan Email Anda untuk Git (contoh: john@example.com): "
    if not "!GIT_USER!"=="" "!GIT_CMD!" config user.name "!GIT_USER!"
    if not "!GIT_EMAIL!"=="" "!GIT_CMD!" config user.email "!GIT_EMAIL!"
)

echo.
echo [+] Menambahkan file proyek (dataset dan runs diabaikan oleh .gitignore)...
"!GIT_CMD!" add .

echo.
echo [+] Membuat initial commit...
"!GIT_CMD!" commit -m "feat: Phytoplankton detection with YOLO26 for 7 selected classes"

echo.
echo [+] Mengatur branch utama ke 'main'...
"!GIT_CMD!" branch -M main

echo.
echo ============================================================
echo   HUBUNGKAN KE GITHUB REMOTE
echo ============================================================
echo Buka https://github.com/new dan buat repository baru (misal: phytoplankton_detection).
echo Jangan centang 'Add a README file' atau '.gitignore' (karena sudah ada).
echo.
set /p REPO_URL="Masukkan URL Repository GitHub Anda (contoh: https://github.com/username/phytoplankton_detection.git): "

if "!REPO_URL!"=="" (
    echo.
    echo [INFO] URL tidak dimasukkan. Anda dapat menghubungkan remote nanti dengan perintah:
    echo        git remote add origin https://github.com/USERNAME/REPO_NAME.git
    echo        git push -u origin main
) else (
    echo.
    echo [+] Menambahkan remote origin: !REPO_URL!
    "!GIT_CMD!" remote remove origin >nul 2>nul
    "!GIT_CMD!" remote add origin !REPO_URL!
    echo.
    echo [+] Mendorong kode ke GitHub (git push -u origin main)...
    "!GIT_CMD!" push -u origin main
    if %errorlevel% equ 0 (
        echo.
        echo ============================================================
        echo [SUKSES] Repositori berhasil di-push ke GitHub!
        echo ============================================================
    ) else (
        echo.
        echo [WARNING] Push belum berhasil. Pastikan Anda sudah login Git/GitHub credential.
        echo Anda dapat mencoba push manual kapan saja dengan:
        echo   git push -u origin main
    )
)

echo.
pause
