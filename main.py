#!/usr/bin/env python3
"""
Image Verification 主程序
演示如何使用验证码识别框架
"""

import sys
import os
from pathlib import Path
import json

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import CaptchaProcessor, CaptchaDetector
from src.core.captcha_detector import CaptchaType
from src.utils import ImageUtils, get_logger

def main():
    """主函数"""
    logger = get_logger("main")
    logger.info("=== Image Verification 验证码识别系统 ===")
    
    # 初始化处理器
    processor = CaptchaProcessor()
    
    # 获取example文件夹中的图像
    example_dir = Path("example")
    if not example_dir.exists():
        logger.error("example文件夹不存在")
        return
    
    # 处理所有示例图像
    image_files = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
    
    if not image_files:
        logger.warning("example文件夹中没有找到图像文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个示例图像")
    
    results = []
    
    for image_file in image_files:
        logger.info(f"\n处理图像: {image_file.name}")
        logger.info("-" * 50)
        
        try:
            # 处理验证码
            result = processor.process_captcha(
                image_file,
                additional_info={'filename': image_file.name}
            )
            
            # 显示结果
            display_result(result, image_file.name)
            results.append({
                'filename': image_file.name,
                'result': result
            })
            
        except Exception as e:
            logger.error(f"处理 {image_file.name} 时出错: {e}")
    
    # 保存结果
    save_results(results)
    
    logger.info("\n=== 处理完成 ===")

def display_result(result: dict, filename: str):
    """显示处理结果"""
    logger = get_logger("display")
    
    logger.info(f"文件: {filename}")
    logger.info(f"检测类型: {result.get('captcha_type', 'unknown')}")
    logger.info(f"检测置信度: {result.get('detection_confidence', 0):.3f}")
    logger.info(f"处理成功: {result.get('success', False)}")
    
    if result.get('success'):
        captcha_type = result.get('captcha_type')
        
        if captcha_type == 'char_captcha':
            text = result.get('text', '')
            confidence = result.get('confidence', 0)
            method = result.get('method', '')
            logger.info(f"识别文本: '{text}' (置信度: {confidence:.1f}, 方法: {method})")
            
        elif captcha_type == 'slider_captcha':
            position = result.get('position', {})
            distance = result.get('distance', 0)
            method = result.get('method', '')
            logger.info(f"滑块位置: ({position.get('x', 0)}, {position.get('y', 0)})")
            logger.info(f"滑动距离: {distance}px (方法: {method})")
            
        elif captcha_type == 'click_captcha':
            targets = result.get('targets', [])
            target_count = result.get('target_count', 0)
            logger.info(f"检测到 {target_count} 个目标")
            for i, target in enumerate(targets[:3]):  # 只显示前3个
                center = target.get('center', (0, 0))
                logger.info(f"  目标{i+1}: 中心点({center[0]}, {center[1]})")
                
        elif captcha_type == 'puzzle_captcha':
            if 'position' in result:
                position = result.get('position', {})
                logger.info(f"拼图位置: ({position.get('x', 0)}, {position.get('y', 0)})")
            else:
                area = result.get('area', 0)
                logger.info(f"拼图面积: {area:.0f}")
    else:
        error = result.get('error', '未知错误')
        logger.error(f"处理失败: {error}")

def save_results(results: list):
    """保存结果到文件"""
    logger = get_logger("save")
    
    # 创建results目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 保存JSON结果
    output_file = results_dir / "captcha_results.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def demo_specific_types():
    """演示特定类型的验证码处理"""
    logger = get_logger("demo")
    processor = CaptchaProcessor()
    
    logger.info("\n=== 特定类型处理演示 ===")
    
    # 演示字符验证码
    example_dir = Path("example")
    image_files = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
    
    if image_files:
        test_image = image_files[0]
        logger.info(f"\n使用 {test_image.name} 演示不同处理方法:")
        
        # 加载图像
        image = ImageUtils.load_image(test_image)
        
        # 强制指定为字符验证码
        result = processor.process_captcha(image, captcha_type=CaptchaType.CHAR_CAPTCHA)
        logger.info(f"强制字符识别: {result.get('text', 'N/A')}")
        
        # 强制指定为滑块验证码
        result = processor.process_captcha(image, captcha_type=CaptchaType.SLIDER_CAPTCHA)
        logger.info(f"强制滑块检测: 距离 {result.get('distance', 0)}px")

if __name__ == "__main__":
    main()
    demo_specific_types() 