#!/usr/bin/env python3
"""
分析example文件夹中的验证码图像
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import ImageUtils, get_logger

def analyze_image(image_path: Path):
    """分析单个图像"""
    logger = get_logger("analyze")
    
    try:
        # 加载图像
        image = ImageUtils.load_image(image_path)
        h, w = image.shape[:2]
        
        logger.info(f"\n=== 分析图像: {image_path.name} ===")
        logger.info(f"尺寸: {w} x {h}")
        logger.info(f"宽高比: {w/h:.2f}")
        logger.info(f"颜色通道: {image.shape[2] if len(image.shape) == 3 else 1}")
        
        # 计算基本特征
        gray = ImageUtils.to_grayscale(image)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        logger.info(f"边缘密度: {edge_density:.4f}")
        
        # 查找轮廓
        contours = ImageUtils.find_contours(edges)
        logger.info(f"轮廓数量: {len(contours)}")
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            logger.info(f"平均轮廓面积: {np.mean(areas):.1f}")
            logger.info(f"最大轮廓面积: {np.max(areas):.1f}")
        
        # 颜色复杂度
        if len(image.shape) == 3:
            colors = image.reshape(-1, image.shape[-1])
            unique_colors = len(np.unique(colors, axis=0))
            color_complexity = unique_colors / (w * h)
            logger.info(f"颜色复杂度: {color_complexity:.4f}")
        
        # 基于特征推测验证码类型
        predict_captcha_type(w, h, edge_density, len(contours), image_path.name)
        
    except Exception as e:
        logger.error(f"分析 {image_path.name} 失败: {e}")

def predict_captcha_type(width: int, height: int, edge_density: float, 
                        contour_count: int, filename: str):
    """基于特征预测验证码类型"""
    logger = get_logger("predict")
    
    aspect_ratio = width / height
    
    predictions = []
    
    # 字符验证码特征
    if (2.0 <= aspect_ratio <= 4.0 and 
        width <= 200 and height <= 100 and
        0.05 <= edge_density <= 0.3 and
        contour_count <= 20):
        predictions.append("字符验证码")
    
    # 滑块验证码特征
    if (aspect_ratio > 2.0 and
        width > 250 and 
        edge_density > 0.1):
        predictions.append("滑块验证码")
    
    # 点击验证码特征
    if (0.8 <= aspect_ratio <= 1.2 and
        contour_count > 10):
        predictions.append("点击验证码")
    
    # 基于文件名
    filename_lower = filename.lower()
    if 'slider' in filename_lower or 'slide' in filename_lower:
        predictions.append("滑块验证码 (文件名)")
    elif 'click' in filename_lower or 'select' in filename_lower:
        predictions.append("点击验证码 (文件名)")
    elif 'char' in filename_lower or 'text' in filename_lower:
        predictions.append("字符验证码 (文件名)")
    
    if predictions:
        logger.info(f"可能的类型: {', '.join(predictions)}")
    else:
        logger.info("未知验证码类型")

def main():
    """主函数"""
    logger = get_logger("main")
    logger.info("=== 验证码图像分析工具 ===")
    
    example_dir = Path("example")
    if not example_dir.exists():
        logger.error("example文件夹不存在")
        return
    
    # 获取所有图像文件
    image_files = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
    
    if not image_files:
        logger.warning("example文件夹中没有找到图像文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 分析每个图像
    for image_file in sorted(image_files):
        analyze_image(image_file)
    
    logger.info("\n=== 分析完成 ===")

if __name__ == "__main__":
    main() 