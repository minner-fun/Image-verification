"""
极验4代点选验证码测试脚本
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src import CaptchaProcessor
from src.core.captcha_detector import CaptchaType

def test_geetest_click_captcha():
    """测试极验4代点选验证码"""
    
    # 初始化处理器
    processor = CaptchaProcessor()
    
    # 定义文件路径
    main_image_path = "example/pic1.jpg"
    template_paths = [
        "example/icon1.png",
        "example/icon2.png", 
        "example/icon3.png"
    ]
    
    # 检查文件是否存在
    if not os.path.exists(main_image_path):
        print(f"❌ 主图像文件不存在: {main_image_path}")
        return
    
    missing_templates = [path for path in template_paths if not os.path.exists(path)]
    if missing_templates:
        print(f"❌ 模板图像文件不存在: {missing_templates}")
        return
    
    print("🎯 开始极验4代点选验证码测试")
    print(f"📷 主图像: {main_image_path}")
    print(f"🔍 模板图像: {template_paths}")
    
    # 加载主图像
    main_image = cv2.imread(main_image_path)
    if main_image is None:
        print(f"❌ 无法加载主图像: {main_image_path}")
        return
    
    print(f"✅ 主图像尺寸: {main_image.shape}")
    
    # 加载并显示模板信息
    for i, template_path in enumerate(template_paths):
        template = cv2.imread(template_path)
        if template is not None:
            print(f"✅ 模板{i+1} ({os.path.basename(template_path)}): {template.shape}")
        else:
            print(f"❌ 无法加载模板{i+1}: {template_path}")
    
    # 使用模板匹配处理
    print("\n🔄 开始模板匹配...")
    result = processor.process_captcha(
        main_image_path,
        captcha_type=CaptchaType.CLICK_CAPTCHA,
        template_paths=template_paths
    )
    
    # 显示结果
    print("\n📊 处理结果:")
    print(f"✅ 成功: {result.get('success', False)}")
    print(f"🎯 方法: {result.get('method', 'unknown')}")
    print(f"📋 目标数量: {result.get('target_count', 0)}")
    print(f"🔧 模板数量: {result.get('template_count', 0)}")
    
    if result.get('success') and result.get('targets'):
        print("\n🎯 找到的目标位置:")
        for i, target in enumerate(result['targets']):
            template_name = target.get('template_name', f'template_{i}')
            center = target.get('center', (0, 0))
            confidence = target.get('confidence', 0)
            
            print(f"  {i+1}. {template_name}")
            print(f"     📍 位置: {center}")
            print(f"     🎯 置信度: {confidence:.3f}")
            print(f"     📦 边界框: {target.get('bbox', 'N/A')}")
    
    # 可视化结果
    create_visualization(main_image, result, template_paths)
    
    return result

def create_visualization(main_image: np.ndarray, result: dict, template_paths: list):
    """创建可视化结果"""
    
    # 复制图像用于绘制
    vis_image = main_image.copy()
    
    if result.get('success') and result.get('targets'):
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 绿、蓝、红
        
        for i, target in enumerate(result['targets']):
            center = target.get('center', (0, 0))
            bbox = target.get('bbox', None)
            confidence = target.get('confidence', 0)
            template_name = target.get('template_name', f'template_{i}')
            
            color = colors[i % len(colors)]
            
            # 绘制中心点
            cv2.circle(vis_image, center, 10, color, -1)
            
            # 绘制边界框
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # 绘制标签
                label = f"{template_name}: {confidence:.2f}"
                cv2.putText(vis_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 保存结果
    output_path = "results/geetest_click_result.jpg"
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    print(f"\n💾 可视化结果已保存: {output_path}")
    
    # 显示图像尺寸信息
    print(f"📐 可视化图像尺寸: {vis_image.shape}")

def analyze_templates():
    """分析模板图像特征"""
    template_paths = [
        "example/icon1.png",
        "example/icon2.png", 
        "example/icon3.png"
    ]
    
    print("\n🔍 模板图像分析:")
    
    for i, template_path in enumerate(template_paths):
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
            if template is not None:
                h, w = template.shape[:2]
                print(f"\n📋 {os.path.basename(template_path)}:")
                print(f"   📐 尺寸: {w}x{h}")
                print(f"   🎨 通道: {template.shape[2] if len(template.shape) > 2 else 1}")
                
                # 计算简单特征
                gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray)
                std_intensity = np.std(gray)
                
                print(f"   💡 平均亮度: {mean_intensity:.2f}")
                print(f"   📊 亮度标准差: {std_intensity:.2f}")

if __name__ == "__main__":
    print("🚀 极验4代点选验证码识别测试")
    print("=" * 50)
    
    # 分析模板
    analyze_templates()
    
    print("\n" + "=" * 50)
    
    # 执行测试
    result = test_geetest_click_captcha()
    
    if result:
        print("\n✅ 测试完成!")
        if result.get('success'):
            print("🎉 成功识别目标位置!")
        else:
            print("⚠️  未能成功识别，可能需要调整参数")
    else:
        print("\n❌ 测试失败!")