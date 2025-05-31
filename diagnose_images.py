"""
图像加载诊断脚本
"""
import cv2
import numpy as np
import os
from PIL import Image

def diagnose_image(image_path):
    """诊断单个图像文件"""
    print(f"\n🔍 诊断图像: {image_path}")
    print("-" * 40)
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print("❌ 文件不存在")
        return None
    
    # 文件大小
    file_size = os.path.getsize(image_path)
    print(f"📁 文件大小: {file_size} bytes")
    
    # 使用OpenCV读取
    print("\n📷 OpenCV读取:")
    img_cv = cv2.imread(image_path)
    if img_cv is not None:
        print(f"   ✅ 成功读取，形状: {img_cv.shape}")
        print(f"   📊 数据类型: {img_cv.dtype}")
        print(f"   📈 像素范围: [{img_cv.min()}, {img_cv.max()}]")
        print(f"   💡 平均值: {img_cv.mean():.2f}")
        print(f"   📏 标准差: {img_cv.std():.2f}")
        
        # 检查是否全黑
        if img_cv.max() == 0:
            print("   ⚠️  图像全黑!")
    else:
        print("   ❌ OpenCV读取失败")
    
    # 使用OpenCV读取（包含alpha通道）
    print("\n📷 OpenCV读取（含alpha）:")
    img_cv_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_cv_alpha is not None:
        print(f"   ✅ 成功读取，形状: {img_cv_alpha.shape}")
        print(f"   📊 数据类型: {img_cv_alpha.dtype}")
        print(f"   📈 像素范围: [{img_cv_alpha.min()}, {img_cv_alpha.max()}]")
        
        if len(img_cv_alpha.shape) == 3 and img_cv_alpha.shape[2] == 4:
            print("   🔄 检测到alpha通道")
            # 分析alpha通道
            alpha = img_cv_alpha[:, :, 3]
            print(f"   🎭 Alpha范围: [{alpha.min()}, {alpha.max()}]")
            print(f"   🎭 Alpha平均: {alpha.mean():.2f}")
            
            # 分析RGB通道
            rgb = img_cv_alpha[:, :, :3]
            print(f"   🌈 RGB平均: {rgb.mean():.2f}")
            print(f"   🌈 RGB标准差: {rgb.std():.2f}")
    else:
        print("   ❌ OpenCV读取失败")
    
    # 使用PIL读取
    print("\n📷 PIL读取:")
    try:
        img_pil = Image.open(image_path)
        print(f"   ✅ 成功读取，尺寸: {img_pil.size}")
        print(f"   🎨 模式: {img_pil.mode}")
        print(f"   📋 格式: {img_pil.format}")
        
        # 转换为numpy数组
        img_array = np.array(img_pil)
        print(f"   📊 数组形状: {img_array.shape}")
        print(f"   📈 像素范围: [{img_array.min()}, {img_array.max()}]")
        print(f"   💡 平均值: {img_array.mean():.2f}")
        
        # 如果有alpha通道，分析透明度
        if img_pil.mode in ['RGBA', 'LA']:
            print("   🔄 检测到透明通道")
            if img_pil.mode == 'RGBA':
                alpha_channel = img_array[:, :, 3]
            else:  # LA
                alpha_channel = img_array[:, :, 1]
            
            print(f"   🎭 透明度范围: [{alpha_channel.min()}, {alpha_channel.max()}]")
            print(f"   🎭 透明度平均: {alpha_channel.mean():.2f}")
            
            # 检查是否完全透明
            if alpha_channel.max() == 0:
                print("   ⚠️  图像完全透明!")
            
            # 转换为RGB看看
            if img_pil.mode == 'RGBA':
                img_rgb = img_pil.convert('RGB')
                rgb_array = np.array(img_rgb)
                print(f"   🌈 转换为RGB后平均值: {rgb_array.mean():.2f}")
        
    except Exception as e:
        print(f"   ❌ PIL读取失败: {e}")
    
    return img_cv, img_cv_alpha

def analyze_all_images():
    """分析所有图像"""
    print("🔍 图像加载问题诊断")
    print("=" * 50)
    
    # 图像路径
    images = [
        "example/pic1.jpg",
        "example/icon1.png", 
        "example/icon2.png",
        "example/icon3.png"
    ]
    
    results = {}
    
    for img_path in images:
        result = diagnose_image(img_path)
        results[img_path] = result
    
    # 总结分析
    print("\n📊 总结分析:")
    print("=" * 50)
    
    for img_path, (img_cv, img_cv_alpha) in results.items():
        if img_cv is not None or img_cv_alpha is not None:
            print(f"✅ {img_path}: 可读取")
            if img_cv is not None and img_cv.max() == 0:
                print(f"   ⚠️  疑似全黑图像")
            if img_cv_alpha is not None and len(img_cv_alpha.shape) == 3 and img_cv_alpha.shape[2] == 4:
                print(f"   🔄 含有alpha通道")
        else:
            print(f"❌ {img_path}: 无法读取")

def test_image_loading_solutions():
    """测试图像加载解决方案"""
    print("\n🔧 测试图像加载解决方案:")
    print("=" * 50)
    
    template_paths = [
        "example/icon1.png",
        "example/icon2.png", 
        "example/icon3.png"
    ]
    
    for template_path in template_paths:
        print(f"\n🧪 测试 {template_path}:")
        
        # 方法1: 直接OpenCV读取
        img1 = cv2.imread(template_path)
        if img1 is not None:
            print(f"   方法1 (直接读取): 形状={img1.shape}, 平均值={img1.mean():.2f}")
        
        # 方法2: OpenCV读取含alpha
        img2 = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if img2 is not None:
            print(f"   方法2 (含alpha): 形状={img2.shape}, 平均值={img2.mean():.2f}")
            
            # 如果有alpha通道，去除alpha
            if len(img2.shape) == 3 and img2.shape[2] == 4:
                img2_rgb = img2[:, :, :3]
                print(f"   方法2a (去alpha): 形状={img2_rgb.shape}, 平均值={img2_rgb.mean():.2f}")
        
        # 方法3: PIL转换
        try:
            from PIL import Image
            img_pil = Image.open(template_path)
            if img_pil.mode in ['RGBA', 'LA']:
                img_pil = img_pil.convert('RGB')
            img3 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            print(f"   方法3 (PIL转换): 形状={img3.shape}, 平均值={img3.mean():.2f}")
        except Exception as e:
            print(f"   方法3 失败: {e}")

if __name__ == "__main__":
    analyze_all_images()
    test_image_loading_solutions()