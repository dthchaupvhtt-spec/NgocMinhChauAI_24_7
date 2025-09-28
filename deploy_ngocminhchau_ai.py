import os
import subprocess
import webbrowser

print('--- NgọcMinhChâu AI 1-Click Deploy Script ---')

# Bước 1: Kiểm tra Python
try:
    python_version = subprocess.check_output(['python', '--version']).decode().strip()
    print(f'Python detected: {python_version}')
except Exception as e:
    print('Python not found. Please install Python 3.10+ before running this script.')
    exit()

# Bước 2: Tạo virtual environment
if not os.path.exists('venv'):
    print('Creating virtual environment...')
    subprocess.run(['python', '-m', 'venv', 'venv'])
else:
    print('Virtual environment already exists.')

# Bước 3: Hướng dẫn kích hoạt và cài thư viện
activate_script = 'venv\\Scripts\\activate' if os.name == 'nt' else 'source venv/bin/activate'
print(f'Activate virtual environment: {activate_script}')
print('After activation, run: pip install -r requirements.txt')

# Bước 4: Git init & commit
if not os.path.exists('.git'):
    print('Initializing Git repository...')
    subprocess.run(['git', 'init'])
subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', 'Initial commit for NgocMinhChau AI'])

# Bước 5: Push lên GitHub
repo_url = input('Enter your GitHub repository URL: ')
subprocess.run(['git', 'remote', 'add', 'origin', repo_url])
subprocess.run(['git', 'branch', '-M', 'main'])
subprocess.run(['git', 'push', '-u', 'origin', 'main'])

# Bước 6: Hướng dẫn deploy trên Render
print('\n--- NEXT STEPS ---')
print('1. Go to https://render.com and login.')
print('2. Create New Web Service → Connect GitHub repository.')
print('3. Branch: main')
print('4. Build Command: pip install -r requirements.txt')
print('5. Start Command: streamlit run app.py --server.port $PORT --server.enableCORS false')
print('6. Add Environment Variables: OPENAI_API_KEY, EMAIL_USER, EMAIL_PASSWORD, ZALO_ACCESS_TOKEN, ZALO_USER_ID')
print('7. Click Deploy. After deploy, you will have your Cloud URL.')

# Bước 7: Mở URL Cloud tự động
cloud_url = input('Enter your deployed Render Cloud URL (after deploy): ')
webbrowser.open(cloud_url)
print('NgọcMinhChâu AI 24/7 is now accessible!')
print('--- Done! ---')
