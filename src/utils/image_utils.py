"""
图像处理工具
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional
from pathlib import Path

class ImageUtils:
    """图像处理工具类"""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy数组格式的图像
        """
        image_path = str(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def save_image(image: np.ndarray, save_path: Union[str, Path], 
                   format: str = 'RGB') -> bool:
        """
        保存图像
        
        Args:
            image: numpy数组格式的图像
            save_path: 保存路径
            format: 图像格式 ('RGB' 或 'BGR')
            
        Returns:
            是否保存成功
        """
        try:
            save_path = str(save_path)
            if format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image)
            return True
        except Exception as e:
            print(f"保存图像失败: {e}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int = None, 
                    height: int = None, keep_aspect: bool = True) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            width: 目标宽度
            height: 目标高度
            keep_aspect: 是否保持宽高比
            
        Returns:
            调整大小后的图像
        """
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        if keep_aspect:
            if width is not None and height is not None:
                # 计算缩放比例
                ratio = min(width/w, height/h)
                new_w, new_h = int(w * ratio), int(h * ratio)
            elif width is not None:
                ratio = width / w
                new_w, new_h = width, int(h * ratio)
            else:
                ratio = height / h
                new_w, new_h = int(w * ratio), height
        else:
            new_w = width or w
            new_h = height or h
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        转换为灰度图
        
        Args:
            image: 输入图像
            
        Returns:
            灰度图像
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, 
                           kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
        """
        应用高斯模糊
        
        Args:
            image: 输入图像
            kernel_size: 核大小
            
        Returns:
            模糊后的图像
        """
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    @staticmethod
    def apply_threshold(image: np.ndarray, threshold: int = 127, 
                       max_value: int = 255, 
                       threshold_type: int = cv2.THRESH_BINARY) -> np.ndarray:
        """
        应用阈值处理
        
        Args:
            image: 输入灰度图像
            threshold: 阈值
            max_value: 最大值
            threshold_type: 阈值类型
            
        Returns:
            二值化图像
        """
        _, binary = cv2.threshold(image, threshold, max_value, threshold_type)
        return binary
    
    @staticmethod
    def apply_morphology(image: np.ndarray, operation: str = 'close',
                        kernel_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        应用形态学操作
        
        Args:
            image: 输入二值图像
            operation: 操作类型 ('open', 'close', 'erode', 'dilate')
            kernel_size: 核大小
            
        Returns:
            处理后的图像
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        operations = {
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'erode': cv2.MORPH_ERODE,
            'dilate': cv2.MORPH_DILATE
        }
        
        if operation not in operations:
            raise ValueError(f"不支持的形态学操作: {operation}")
        
        return cv2.morphologyEx(image, operations[operation], kernel)
    
    @staticmethod
    def find_contours(image: np.ndarray) -> list:
        """
        查找轮廓
        
        Args:
            image: 输入二值图像
            
        Returns:
            轮廓列表
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def crop_image(image: np.ndarray, x: int, y: int, 
                  width: int, height: int) -> np.ndarray:
        """
        裁剪图像
        
        Args:
            image: 输入图像
            x: 起始x坐标
            y: 起始y坐标  
            width: 宽度
            height: 高度
            
        Returns:
            裁剪后的图像
        """
        return image[y:y+height, x:x+width]
    
    @staticmethod
    def template_match(image: np.ndarray, template: np.ndarray,
                      method: int = cv2.TM_CCOEFF_NORMED) -> Tuple[float, Tuple[int, int]]:
        """
        模板匹配
        
        Args:
            image: 目标图像
            template: 模板图像
            method: 匹配方法
            
        Returns:
            (最佳匹配值, (x, y)坐标)
        """
        result = cv2.matchTemplate(image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            return min_val, min_loc
        else:
            return max_val, max_loc 