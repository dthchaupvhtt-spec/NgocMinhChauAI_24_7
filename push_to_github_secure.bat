@echo off
echo ===========================================
echo   WinX - Push NgocMinhChauAI len GitHub (Secure)
echo ===========================================

:: Thong tin GitHub (thay bang thong tin cua Minh)
set GITHUB_USER=dthchaupvhtt-spec
set GITHUB_REPO=NgocMinhChauAI_24_7
set /p GITHUB_TOKEN=Nhap GitHub token (PAT): 

:: Tao file .gitignore neu chua co
if not exist .gitignore (
  echo venv/>>.gitignore
  echo __pycache__/>>.gitignore
  echo *.pyc>>.gitignore
  echo *.pyo>>.gitignore
  echo *.pyd>>.gitignore
  echo *.log>>.gitignore
  echo *.sqlite3>>.gitignore
  echo .env>>.gitignore
)

:: Cau hinh Git
git config --global user.name "Minh"
git config --global user.email "your_email@example.com"

:: Dam bao .env khong bi push
git rm --cached .env

:: Khoi tao Git repo neu chua co
git init
git add .
git commit -m "Clean commit - NgocMinhChauAI"
git branch -M main

:: Xoa origin cu (neu co) va them origin moi
git remote remove origin
git remote add origin https://%GITHUB_TOKEN%@github.com/%GITHUB_USER%/%GITHUB_REPO%.git

:: Push code
git push -u origin main --force

echo ===========================================
echo   Da push code len GitHub thanh cong!
echo ===========================================
pause
