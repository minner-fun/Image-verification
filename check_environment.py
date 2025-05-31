#!/usr/bin/env python3
"""
检查环境配置脚本
验证GPU、CUDA、PyTorch等依赖是否正确安装
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print(f"🐍 Python版本: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  警告: 推荐使用Python 3.8+")
    else:
        print("✅ Python版本符合要求")

def check_cuda():
    """检查CUDA版本"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🔥 NVIDIA GPU检测:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version: ')[1].split()[0]
                    print(f"   CUDA版本: {cuda_version}")
                    break
            print("✅ NVIDIA GPU可用")
        else:
            print("❌ 未检测到NVIDIA GPU")
    except FileNotFoundError:
        print("❌ nvidia-smi命令未找到，可能没有安装NVIDIA驱动")

def check_pytorch():
    """检查PyTorch安装"""
    try:
        import torch
        print(f"\n🔥 PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            print("✅ PyTorch GPU支持正常")
        else:
            print("⚠️  PyTorch未检测到CUDA支持")
    except ImportError:
        print("❌ PyTorch未安装")

def check_opencv():
    """检查OpenCV安装"""
    try:
        import cv2
        print(f"\n📷 OpenCV版本: {cv2.__version__}")
        
        # 检查GPU支持
        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0:
                print(f"   GPU设备数量: {gpu_count}")
                print("✅ OpenCV GPU支持可用")
            else:
                print("⚠️  OpenCV未检测到GPU支持")
        except:
            print("⚠️  OpenCV没有CUDA支持")
            
    except ImportError:
        print("❌ OpenCV未安装")

def check_ocr_engines():
    """检查OCR引擎"""
    print("\n🔤 OCR引擎检查:")
    
    # 检查Tesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"   Tesseract版本: {version}")
        print("✅ Tesseract可用")
    except:
        print("❌ Tesseract不可用")
    
    # 检查EasyOCR
    try:
        import easyocr
        print("✅ EasyOCR已安装")
        
        # 测试GPU支持
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            print("✅ EasyOCR GPU支持可用")
        except:
            print("⚠️  EasyOCR GPU支持不可用，将使用CPU")
            
    except ImportError:
        print("❌ EasyOCR未安装")

def check_yolo():
    """检查YOLO/Ultralytics"""
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"\n🎯 Ultralytics版本: {ultralytics.__version__}")
        
        # 测试YOLO模型加载
        try:
            model = YOLO('yolov8n.pt')  # 会自动下载
            print("✅ YOLO模型加载成功")
        except Exception as e:
            print(f"⚠️  YOLO模型加载失败: {e}")
            
    except ImportError:
        print("❌ Ultralytics未安装")

def check_dependencies():
    """检查其他依赖"""
    print("\n📦 其他依赖检查:")
    
    deps = {
        'numpy': 'NumPy',
        'pandas': 'Pandas', 
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'loguru': 'Loguru',
        'yaml': 'PyYAML'
    }
    
    for module, name in deps.items():
        try:
            if module == 'yaml':
                import yaml
            else:
                __import__(module)
            print(f"✅ {name}已安装")
        except ImportError:
            print(f"❌ {name}未安装")

def check_project_structure():
    """检查项目结构"""
    print("\n📁 项目结构检查:")
    
    required_dirs = ['src', 'config', 'example']
    required_files = ['main.py', 'analyze_examples.py', 'requirements.txt']
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ 目录 {dir_name}/ 存在")
        else:
            print(f"❌ 目录 {dir_name}/ 不存在")
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ 文件 {file_name} 存在")
        else:
            print(f"❌ 文件 {file_name} 不存在")

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 Image Verification 环境检查")
    print("=" * 60)
    
    check_python_version()
    check_cuda()
    check_pytorch()
    check_opencv()
    check_ocr_engines()
    check_yolo()
    check_dependencies()
    check_project_structure()
    
    print("\n" + "=" * 60)
    print("🎉 环境检查完成！")
    print("=" * 60)
    
    print("\n💡 安装建议:")
    print("1. 如果PyTorch没有GPU支持，运行:")
    print("   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121")
    print("\n2. 如果需要安装完整依赖:")
    print("   pip install -r requirements.txt")
    print("\n3. 安装Tesseract OCR:")
    print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Ubuntu: sudo apt-get install tesseract-ocr")

if __name__ == "__main__":
    main() 