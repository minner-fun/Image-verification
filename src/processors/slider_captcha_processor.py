"""
滑块验证码处理器
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base_processor import BaseProcessor
from ..utils import ImageUtils, config

class SliderCaptchaProcessor(BaseProcessor):
    """滑块验证码处理器"""
    
    def __init__(self):
        super().__init__()
        self.template_threshold = config.get('template_matching.threshold', 0.8)
        self.edge_threshold = config.get('preprocessing.slider_captcha.edge_threshold', [50, 150])
        self.blur_kernel = config.get('preprocessing.slider_captcha.blur_kernel', 5)
    
    def process(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        处理滑块验证码（检测缺口位置）
        
        Args:
            image: 背景图像
            **kwargs: 可能包含 slider_image（滑块图像）
            
        Returns:
            处理结果
        """
        try:
            slider_image = kwargs.get('slider_image')
            
            if slider_image is not None:
                # 如果提供了滑块图像，使用模板匹配
                result = self.process_slider(image, slider_image)
            else:
                # 只有背景图像，检测缺口位置
                result = self.detect_gap(image)
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"滑块验证码处理失败: {str(e)}")
            return self.postprocess_result({
                'success': False,
                'error': str(e)
            })
    
    def process_slider(self, background_image: np.ndarray, 
                      slider_image: np.ndarray) -> Dict[str, Any]:
        """
        处理滑块验证码（模板匹配）
        
        Args:
            background_image: 背景图像
            slider_image: 滑块图像
            
        Returns:
            处理结果
        """
        self.logger.info("开始滑块模板匹配")
        
        # 预处理
        bg_processed = self.preprocess_image(background_image)
        slider_processed = self.preprocess_image(slider_image)
        
        # 转换为灰度图
        bg_gray = ImageUtils.to_grayscale(bg_processed)
        slider_gray = ImageUtils.to_grayscale(slider_processed)
        
        # 模板匹配
        match_value, match_location = ImageUtils.template_match(
            bg_gray, slider_gray, cv2.TM_CCOEFF_NORMED
        )
        
        x, y = match_location
        h, w = slider_gray.shape
        
        result = {
            'method': 'template_matching',
            'match_value': float(match_value),
            'position': {
                'x': int(x),
                'y': int(y)
            },
            'slider_size': {
                'width': int(w),
                'height': int(h)
            },
            'distance': int(x),  # 滑动距离
            'success': match_value >= self.template_threshold
        }
        
        self.logger.info(f"模板匹配完成，位置: ({x}, {y}), 匹配值: {match_value:.3f}")
        return result
    
    def detect_gap(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测缺口位置
        
        Args:
            image: 背景图像
            
        Returns:
            检测结果
        """
        self.logger.info("开始检测缺口位置")
        
        # 预处理
        processed_image = self.preprocess_image(image)
        gray = ImageUtils.to_grayscale(processed_image)
        
        # 边缘检测
        edges = cv2.Canny(gray, self.edge_threshold[0], self.edge_threshold[1])
        
        # 查找轮廓
        contours = ImageUtils.find_contours(edges)
        
        if not contours:
            return {
                'method': 'gap_detection',
                'success': False,
                'error': '未找到轮廓'
            }
        
        # 分析轮廓，寻找缺口特征
        gap_candidates = []
        
        for contour in contours:
            # 计算轮廓属性
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # 缺口特征：
            # 1. 面积适中（不太大不太小）
            # 2. 宽高比接近1:1或略宽
            # 3. 位于图像中央区域
            if (50 < area < 5000 and 
                0.8 < aspect_ratio < 2.0 and
                image.shape[1] * 0.1 < x < image.shape[1] * 0.9):
                
                gap_candidates.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
        
        if not gap_candidates:
            return {
                'method': 'gap_detection',
                'success': False,
                'error': '未找到缺口候选'
            }
        
        # 选择最佳候选（通常是面积最大的）
        best_gap = max(gap_candidates, key=lambda x: x['area'])
        
        result = {
            'method': 'gap_detection',
            'position': {
                'x': best_gap['x'],
                'y': best_gap['y']
            },
            'gap_size': {
                'width': best_gap['width'],
                'height': best_gap['height']
            },
            'distance': best_gap['x'],
            'candidates_count': len(gap_candidates),
            'success': True
        }
        
        self.logger.info(f"缺口检测完成，位置: ({best_gap['x']}, {best_gap['y']})")
        return result
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        滑块验证码图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 高斯模糊去噪
        if self.blur_kernel > 1:
            kernel_size = (self.blur_kernel, self.blur_kernel)
            image = ImageUtils.apply_gaussian_blur(image, kernel_size)
        
        return image
    
    def calculate_slide_path(self, start_x: int, end_x: int, 
                           image_height: int) -> list:
        """
        计算滑动轨迹
        
        Args:
            start_x: 起始x坐标
            end_x: 结束x坐标
            image_height: 图像高度
            
        Returns:
            轨迹点列表 [(x, y), ...]
        """
        distance = end_x - start_x
        y = image_height // 2  # 在图像中央滑动
        
        # 生成更真实的滑动轨迹（模拟人类滑动）
        path = []
        num_steps = max(10, distance // 5)  # 步数与距离成正比
        
        for i in range(num_steps + 1):
            # 使用缓动函数使轨迹更自然
            progress = i / num_steps
            # 使用ease-out曲线
            eased_progress = 1 - (1 - progress) ** 2
            
            x = start_x + int(distance * eased_progress)
            
            # 添加少量随机偏移模拟人类操作
            y_offset = np.random.randint(-2, 3) if i > 0 and i < num_steps else 0
            
            path.append((x, y + y_offset))
        
        return path
    
    def analyze_gap_features(self, image: np.ndarray, 
                           gap_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        分析缺口特征
        
        Args:
            image: 图像
            gap_position: 缺口位置 (x, y)
            
        Returns:
            缺口特征分析结果
        """
        x, y = gap_position
        h, w = image.shape[:2]
        
        # 提取缺口区域
        gap_size = 60  # 假设缺口大小
        x1 = max(0, x - gap_size // 2)
        x2 = min(w, x + gap_size // 2)
        y1 = max(0, y - gap_size // 2)
        y2 = min(h, y + gap_size // 2)
        
        gap_region = image[y1:y2, x1:x2]
        
        if gap_region.size == 0:
            return {'error': '缺口区域为空'}
        
        # 分析缺口特征
        gray_gap = ImageUtils.to_grayscale(gap_region)
        
        features = {
            'mean_intensity': np.mean(gray_gap),
            'std_intensity': np.std(gray_gap),
            'edge_density': np.sum(cv2.Canny(gray_gap, 50, 150) > 0) / gray_gap.size,
            'gap_region_size': gap_region.shape,
            'position': gap_position
        }
        
        return features 