# 12D_Split
12課分割


常用PY指令
python --version
python -m venv env 
.\env\Scripts\Activate.ps1
---------------------------------------------------------------------------------

【12D】

# 切到你專案根目錄
cd C:\Users\wei-kuo\ImageRecognition\Git_12D_Split\12D_Split

# 建立並啟用 venv
pyenv shell 3.10.11
python -m venv env5080
.\env5080\Scripts\Activate.ps1

# 基礎升級
python -m pip install -U pip setuptools wheel

# 先裝 typing-extensions，避開 cu128 metadata 衝突
pip install typing-extensions==4.12.2

# 安裝 PyTorch（CUDA 12.8），不鎖版本（官方建議）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 其他依賴（YOLO 與常用）
pip install ultralytics opencv-python numpy pillow av requests tqdm psutil pyinstaller scikit-image

--------------------------------------------------------------------------------------

# 切到你的正式機專案根目錄
cd C:\Path\To\Your\App

# 建立並啟用 venv
pyenv shell 3.10.11
python -m venv envL4
.\envL4\Scripts\Activate.ps1

# 基礎升級
python -m pip install -U pip setuptools wheel

# 安裝 PyTorch（CUDA 12.6)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 其他依賴（與開發機一致，便於行為對齊）
pip install ultralytics opencv-python numpy pillow av requests tqdm psutil pyinstaller scikit-image   