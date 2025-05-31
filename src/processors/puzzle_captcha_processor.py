"""
拼图验证码处理器
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional
from .base_processor import BaseProcessor
from ..utils import ImageUtils

class PuzzleCaptchaProcessor(BaseProcessor):
    """拼图验证码处理器"""
    
    def __init__(self):
        super().__init__()
        
    def process(self, puzzle_image: np.ndarray, 
                background_image: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        处理拼图验证码
        
        Args:
            puzzle_image: 拼图片图像
            background_image: 背景图像（可选）
            **kwargs: 其他参数
            
        Returns:
            处理结果
        """
        try:
            if background_image is not None:
                # 有背景图的情况，寻找拼图位置
                result = self.find_puzzle_position(puzzle_image, background_image)
            else:
                # 只有拼图片，分析拼图特征
                result = self.analyze_puzzle_shape(puzzle_image)
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"拼图验证码处理失败: {str(e)}")
            return self.postprocess_result({
                'success': False,
                'error': str(e)
            })
    
    def find_puzzle_position(self, puzzle_image: np.ndarray, 
                           background_image: np.ndarray) -> Dict[str, Any]:
        """
        在背景图中找到拼图位置
        
        Args:
            puzzle_image: 拼图片
            background_image: 背景图
            
        Returns:
            位置信息
        """
        self.logger.info("开始寻找拼图位置")
        
        # 转换为灰度图
        puzzle_gray = ImageUtils.to_grayscale(puzzle_image)
        bg_gray = ImageUtils.to_grayscale(background_image)
        
        # 模板匹配
        match_value, match_location = ImageUtils.template_match(
            bg_gray, puzzle_gray, cv2.TM_CCOEFF_NORMED
        )
        
        x, y = match_location
        h, w = puzzle_gray.shape
        
        result = {
            'method': 'template_matching',
            'position': {
                'x': int(x),
                'y': int(y)
            },
            'puzzle_size': {
                'width': int(w),
                'height': int(h)
            },
            'match_value': float(match_value),
            'success': match_value > 0.7
        }
        
        self.logger.info(f"拼图位置: ({x}, {y}), 匹配值: {match_value:.3f}")
        return result
    
    def analyze_puzzle_shape(self, puzzle_image: np.ndarray) -> Dict[str, Any]:
        """
        分析拼图形状特征
        
        Args:
            puzzle_image: 拼图图像
            
        Returns:
            形状特征
        """
        self.logger.info("开始分析拼图形状")
        
        # 转换为灰度图并二值化
        gray = ImageUtils.to_grayscale(puzzle_image)
        binary = ImageUtils.apply_threshold(gray)
        
        # 查找轮廓
        contours = ImageUtils.find_contours(binary)
        
        if not contours:
            return {
                'method': 'shape_analysis',
                'success': False,
                'error': '未找到轮廓'
            }
        
        # 找到最大轮廓（拼图主体）
        main_contour = max(contours, key=cv2.contourArea)
        
        # 分析轮廓特征
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # 计算形状特征
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0
        solidity = area / cv2.contourArea(cv2.convexHull(main_contour)) if cv2.contourArea(cv2.convexHull(main_contour)) > 0 else 0
        
        result = {
            'method': 'shape_analysis',
            'area': float(area),
            'perimeter': float(perimeter),
            'bounding_box': (int(x), int(y), int(w), int(h)),
            'aspect_ratio': float(aspect_ratio),
            'extent': float(extent),
            'solidity': float(solidity),
            'success': True
        }
        
        self.logger.info(f"拼图特征分析完成，面积: {area:.0f}")
        return result 