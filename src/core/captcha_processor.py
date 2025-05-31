"""
验证码处理器主类
"""
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .captcha_detector import CaptchaDetector, CaptchaType
from ..processors.char_captcha_processor import CharCaptchaProcessor
from ..processors.slider_captcha_processor import SliderCaptchaProcessor
from ..processors.click_captcha_processor import ClickCaptchaProcessor
from ..processors.puzzle_captcha_processor import PuzzleCaptchaProcessor
from ..utils import get_logger, ImageUtils

class CaptchaProcessor:
    """验证码处理器主类"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.detector = CaptchaDetector()
        
        # 初始化各类型处理器
        self.processors = {
            CaptchaType.CHAR_CAPTCHA: CharCaptchaProcessor(),
            CaptchaType.SLIDER_CAPTCHA: SliderCaptchaProcessor(),
            CaptchaType.CLICK_CAPTCHA: ClickCaptchaProcessor(),
            CaptchaType.PUZZLE_CAPTCHA: PuzzleCaptchaProcessor()
        }
        
        self.logger.info("验证码处理器初始化完成")
    
    def process_captcha(self, image_input: Union[str, Path, np.ndarray],
                       captcha_type: Optional[CaptchaType] = None,
                       additional_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        处理验证码
        
        Args:
            image_input: 图像输入（文件路径或numpy数组）
            captcha_type: 指定的验证码类型，如果为None则自动检测
            additional_info: 额外信息
            
        Returns:
            处理结果字典
        """
        try:
            # 加载图像
            if isinstance(image_input, (str, Path)):
                image = ImageUtils.load_image(image_input)
                if additional_info is None:
                    additional_info = {}
                additional_info['filename'] = str(image_input)
            else:
                image = image_input
            
            self.logger.info(f"开始处理验证码，图像尺寸: {image.shape}")
            
            # 检测验证码类型（如果未指定）
            if captcha_type is None:
                captcha_type = self.detector.detect_captcha_type(image, additional_info)
                confidence = self.detector.get_detection_confidence(image, captcha_type)
                self.logger.info(f"自动检测到验证码类型: {captcha_type.value}, 置信度: {confidence:.3f}")
            else:
                confidence = 1.0
                self.logger.info(f"使用指定的验证码类型: {captcha_type.value}")
            
            # 获取对应的处理器
            if captcha_type not in self.processors:
                raise ValueError(f"不支持的验证码类型: {captcha_type}")
            
            processor = self.processors[captcha_type]
            
            # 处理验证码
            result = processor.process(image)
            
            # 添加检测信息到结果中
            result.update({
                'captcha_type': captcha_type.value,
                'detection_confidence': confidence,
                'image_shape': image.shape
            })
            
            self.logger.info(f"验证码处理完成，类型: {captcha_type.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"验证码处理失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'captcha_type': captcha_type.value if captcha_type else 'unknown'
            }
    
    def process_char_captcha(self, image: np.ndarray, 
                           ocr_engine: str = 'tesseract') -> Dict[str, Any]:
        """
        处理字符验证码
        
        Args:
            image: 输入图像
            ocr_engine: OCR引擎 ('tesseract' 或 'easyocr')
            
        Returns:
            处理结果
        """
        processor = self.processors[CaptchaType.CHAR_CAPTCHA]
        return processor.process(image, ocr_engine=ocr_engine)
    
    def process_slider_captcha(self, background_image: np.ndarray,
                              slider_image: np.ndarray) -> Dict[str, Any]:
        """
        处理滑块验证码
        
        Args:
            background_image: 背景图像
            slider_image: 滑块图像
            
        Returns:
            处理结果
        """
        processor = self.processors[CaptchaType.SLIDER_CAPTCHA]
        return processor.process_slider(background_image, slider_image)
    
    def process_click_captcha(self, image: np.ndarray,
                             target_description: str = None) -> Dict[str, Any]:
        """
        处理点击验证码
        
        Args:
            image: 输入图像
            target_description: 目标描述（如"点击所有汽车"）
            
        Returns:
            处理结果
        """
        processor = self.processors[CaptchaType.CLICK_CAPTCHA]
        return processor.process(image, target_description=target_description)
    
    def process_puzzle_captcha(self, puzzle_image: np.ndarray,
                              background_image: np.ndarray = None) -> Dict[str, Any]:
        """
        处理拼图验证码
        
        Args:
            puzzle_image: 拼图图像
            background_image: 背景图像（可选）
            
        Returns:
            处理结果
        """
        processor = self.processors[CaptchaType.PUZZLE_CAPTCHA]
        return processor.process(puzzle_image, background_image)
    
    def get_supported_types(self) -> list:
        """
        获取支持的验证码类型列表
        
        Returns:
            支持的类型列表
        """
        return [captcha_type.value for captcha_type in self.processors.keys()]
    
    def update_processor_config(self, captcha_type: CaptchaType, 
                               config: Dict[str, Any]):
        """
        更新处理器配置
        
        Args:
            captcha_type: 验证码类型
            config: 新配置
        """
        if captcha_type in self.processors:
            self.processors[captcha_type].update_config(config)
            self.logger.info(f"已更新 {captcha_type.value} 处理器配置")
        else:
            self.logger.warning(f"不支持的验证码类型: {captcha_type.value}")
    
    def benchmark_detection(self, test_images: list, 
                           ground_truth: list) -> Dict[str, float]:
        """
        基准测试检测准确率
        
        Args:
            test_images: 测试图像列表
            ground_truth: 真实标签列表
            
        Returns:
            性能指标
        """
        if len(test_images) != len(ground_truth):
            raise ValueError("测试图像数量与标签数量不匹配")
        
        correct = 0
        total = len(test_images)
        
        for image, true_type in zip(test_images, ground_truth):
            predicted_type = self.detector.detect_captcha_type(image)
            if predicted_type.value == true_type:
                correct += 1
        
        accuracy = correct / total
        self.logger.info(f"检测准确率: {accuracy:.3f} ({correct}/{total})")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        } 