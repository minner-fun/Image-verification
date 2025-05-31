"""
基础验证码处理器
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
from ..utils import get_logger

class BaseProcessor(ABC):
    """验证码处理器基类"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = {}
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        处理验证码
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            处理结果字典
        """
        pass
    
    def update_config(self, config: Dict[str, Any]):
        """
        更新配置
        
        Args:
            config: 新配置
        """
        self.config.update(config)
        self.logger.info(f"配置已更新: {config}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 子类可以重写此方法实现特定的预处理
        return image
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        结果后处理
        
        Args:
            result: 原始结果
            
        Returns:
            处理后的结果
        """
        # 添加通用信息
        result['processor'] = self.__class__.__name__
        result['success'] = result.get('success', True)
        return result 