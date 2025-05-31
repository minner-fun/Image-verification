"""
验证码类型检测器
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional
from enum import Enum
from ..utils import get_logger, ImageUtils

logger = get_logger(__name__)

class CaptchaType(Enum):
    """验证码类型枚举"""
    CHAR_CAPTCHA = "char_captcha"      # 字符验证码
    SLIDER_CAPTCHA = "slider_captcha"  # 滑块验证码
    CLICK_CAPTCHA = "click_captcha"    # 点击验证码
    PUZZLE_CAPTCHA = "puzzle_captcha"  # 拼图验证码
    UNKNOWN = "unknown"                # 未知类型

class CaptchaDetector:
    """验证码类型检测器"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def detect_captcha_type(self, image: np.ndarray, 
                           additional_info: Optional[Dict] = None) -> CaptchaType:
        """
        检测验证码类型
        
        Args:
            image: 输入图像
            additional_info: 额外信息（如HTML结构、文件名等）
            
        Returns:
            验证码类型
        """
        self.logger.info("开始检测验证码类型")
        
        # 分析图像特征
        features = self._analyze_image_features(image)
        
        # 基于特征判断类型
        captcha_type = self._classify_by_features(features, additional_info)
        
        self.logger.info(f"检测到验证码类型: {captcha_type.value}")
        return captcha_type
    
    def _analyze_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基本尺寸信息
        h, w = image.shape[:2]
        features['width'] = w
        features['height'] = h
        features['aspect_ratio'] = w / h
        
        # 颜色特征
        if len(image.shape) == 3:
            features['is_color'] = True
            features['color_channels'] = image.shape[2]
            
            # 计算颜色复杂度
            colors = image.reshape(-1, image.shape[-1])
            unique_colors = len(np.unique(colors, axis=0))
            features['color_complexity'] = unique_colors / (w * h)
        else:
            features['is_color'] = False
            features['color_channels'] = 1
            features['color_complexity'] = 0
        
        # 边缘特征
        gray = ImageUtils.to_grayscale(image)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (w * h)
        
        # 轮廓特征
        contours = ImageUtils.find_contours(edges)
        features['contour_count'] = len(contours)
        
        if contours:
            # 分析轮廓大小分布
            areas = [cv2.contourArea(c) for c in contours]
            features['avg_contour_area'] = np.mean(areas)
            features['max_contour_area'] = np.max(areas)
            features['contour_area_std'] = np.std(areas)
        else:
            features['avg_contour_area'] = 0
            features['max_contour_area'] = 0
            features['contour_area_std'] = 0
        
        # 纹理特征（基于灰度共生矩阵的简化版本）
        features['texture_variance'] = np.var(gray)
        features['texture_mean'] = np.mean(gray)
        
        return features
    
    def _classify_by_features(self, features: Dict[str, Any], 
                             additional_info: Optional[Dict] = None) -> CaptchaType:
        """
        基于特征分类验证码类型
        
        Args:
            features: 图像特征
            additional_info: 额外信息
            
        Returns:
            验证码类型
        """
        # 检查额外信息中的线索
        if additional_info:
            filename = additional_info.get('filename', '').lower()
            html_content = additional_info.get('html_content', '').lower()
            
            # 基于文件名判断
            if 'slider' in filename or 'slide' in filename:
                return CaptchaType.SLIDER_CAPTCHA
            elif 'click' in filename or 'select' in filename:
                return CaptchaType.CLICK_CAPTCHA
            elif 'puzzle' in filename or 'jigsaw' in filename:
                return CaptchaType.PUZZLE_CAPTCHA
            
            # 基于HTML内容判断
            if 'slider' in html_content or '滑块' in html_content:
                return CaptchaType.SLIDER_CAPTCHA
            elif 'click' in html_content or '点击' in html_content:
                return CaptchaType.CLICK_CAPTCHA
        
        # 基于图像特征判断
        w, h = features['width'], features['height']
        aspect_ratio = features['aspect_ratio']
        edge_density = features['edge_density']
        contour_count = features['contour_count']
        color_complexity = features['color_complexity']
        
        # 字符验证码特征：
        # - 通常较小，宽高比在2:1到4:1之间
        # - 边缘密度适中
        # - 轮廓数量较少（字符数量）
        if (2.0 <= aspect_ratio <= 4.0 and 
            w <= 200 and h <= 100 and
            0.05 <= edge_density <= 0.3 and
            contour_count <= 20 and
            color_complexity < 0.5):
            return CaptchaType.CHAR_CAPTCHA
        
        # 滑块验证码特征：
        # - 通常较大，宽高比较大
        # - 有明显的缺口特征（高边缘密度）
        # - 背景相对复杂
        if (aspect_ratio > 2.0 and
            w > 250 and 
            edge_density > 0.1 and
            color_complexity > 0.3):
            
            # 进一步检查是否有滑块特征
            if self._detect_slider_features(features):
                return CaptchaType.SLIDER_CAPTCHA
        
        # 点击验证码特征：
        # - 通常是正方形或接近正方形
        # - 包含多个小物体（高轮廓数量）
        # - 颜色复杂度高
        if (0.8 <= aspect_ratio <= 1.2 and
            contour_count > 10 and
            color_complexity > 0.4):
            return CaptchaType.CLICK_CAPTCHA
        
        # 拼图验证码特征：
        # - 有明显的拼图形状轮廓
        # - 颜色复杂度高
        if (color_complexity > 0.5 and
            self._detect_puzzle_features(features)):
            return CaptchaType.PUZZLE_CAPTCHA
        
        return CaptchaType.UNKNOWN
    
    def _detect_slider_features(self, features: Dict[str, Any]) -> bool:
        """
        检测滑块特征
        
        Args:
            features: 图像特征
            
        Returns:
            是否包含滑块特征
        """
        # 滑块验证码通常有以下特征：
        # 1. 矩形的缺口
        # 2. 较高的边缘密度
        # 3. 特定的轮廓分布
        
        edge_density = features['edge_density']
        contour_count = features['contour_count']
        avg_contour_area = features['avg_contour_area']
        
        # 简单的启发式规则
        return (edge_density > 0.15 and 
                contour_count > 5 and 
                avg_contour_area > 100)
    
    def _detect_puzzle_features(self, features: Dict[str, Any]) -> bool:
        """
        检测拼图特征
        
        Args:
            features: 图像特征
            
        Returns:
            是否包含拼图特征
        """
        # 拼图验证码通常有复杂的形状轮廓
        contour_count = features['contour_count']
        contour_area_std = features['contour_area_std']
        
        # 拼图片通常有不规则的形状，轮廓面积变化较大
        return (contour_count > 3 and 
                contour_area_std > 1000)
    
    def get_detection_confidence(self, image: np.ndarray, 
                                captcha_type: CaptchaType) -> float:
        """
        获取检测置信度
        
        Args:
            image: 输入图像
            captcha_type: 检测到的类型
            
        Returns:
            置信度分数 (0-1)
        """
        features = self._analyze_image_features(image)
        
        # 基于特征匹配程度计算置信度
        confidence_scores = {
            CaptchaType.CHAR_CAPTCHA: self._char_captcha_confidence(features),
            CaptchaType.SLIDER_CAPTCHA: self._slider_captcha_confidence(features),
            CaptchaType.CLICK_CAPTCHA: self._click_captcha_confidence(features),
            CaptchaType.PUZZLE_CAPTCHA: self._puzzle_captcha_confidence(features)
        }
        
        return confidence_scores.get(captcha_type, 0.0)
    
    def _char_captcha_confidence(self, features: Dict[str, Any]) -> float:
        """计算字符验证码置信度"""
        score = 0.0
        
        # 宽高比评分
        aspect_ratio = features['aspect_ratio']
        if 2.0 <= aspect_ratio <= 4.0:
            score += 0.3
        
        # 尺寸评分
        w, h = features['width'], features['height']
        if w <= 200 and h <= 100:
            score += 0.3
        
        # 边缘密度评分
        edge_density = features['edge_density']
        if 0.05 <= edge_density <= 0.3:
            score += 0.2
        
        # 轮廓数量评分
        contour_count = features['contour_count']
        if contour_count <= 20:
            score += 0.2
        
        return min(score, 1.0)
    
    def _slider_captcha_confidence(self, features: Dict[str, Any]) -> float:
        """计算滑块验证码置信度"""
        score = 0.0
        
        # 宽高比评分
        aspect_ratio = features['aspect_ratio']
        if aspect_ratio > 2.0:
            score += 0.3
        
        # 尺寸评分
        w = features['width']
        if w > 250:
            score += 0.2
        
        # 边缘密度评分
        edge_density = features['edge_density']
        if edge_density > 0.1:
            score += 0.3
        
        # 颜色复杂度评分
        color_complexity = features['color_complexity']
        if color_complexity > 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _click_captcha_confidence(self, features: Dict[str, Any]) -> float:
        """计算点击验证码置信度"""
        score = 0.0
        
        # 宽高比评分（接近正方形）
        aspect_ratio = features['aspect_ratio']
        if 0.8 <= aspect_ratio <= 1.2:
            score += 0.4
        
        # 轮廓数量评分
        contour_count = features['contour_count']
        if contour_count > 10:
            score += 0.3
        
        # 颜色复杂度评分
        color_complexity = features['color_complexity']
        if color_complexity > 0.4:
            score += 0.3
        
        return min(score, 1.0)
    
    def _puzzle_captcha_confidence(self, features: Dict[str, Any]) -> float:
        """计算拼图验证码置信度"""
        score = 0.0
        
        # 颜色复杂度评分
        color_complexity = features['color_complexity']
        if color_complexity > 0.5:
            score += 0.4
        
        # 轮廓特征评分
        contour_count = features['contour_count']
        contour_area_std = features['contour_area_std']
        
        if contour_count > 3:
            score += 0.3
        
        if contour_area_std > 1000:
            score += 0.3
        
        return min(score, 1.0) 