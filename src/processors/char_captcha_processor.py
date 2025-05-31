"""
字符验证码处理器
"""
import cv2
import numpy as np
from typing import Dict, Any
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from .base_processor import BaseProcessor
from ..utils import ImageUtils, config

class CharCaptchaProcessor(BaseProcessor):
    """字符验证码处理器"""
    
    def __init__(self):
        super().__init__()
        self.tesseract_config = config.get('ocr.tesseract.config', '--oem 3 --psm 6')
        self.tesseract_lang = config.get('ocr.tesseract.language', 'eng')
        
        # EasyOCR配置
        self.easyocr_langs = config.get('ocr.easyocr.languages', ['en'])
        self.easyocr_gpu = config.get('ocr.easyocr.gpu', False)
        
        # 初始化EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(self.easyocr_langs, gpu=self.easyocr_gpu)
            except Exception as e:
                self.logger.warning(f"EasyOCR初始化失败: {e}")
                self.easyocr_reader = None
        else:
            self.easyocr_reader = None
    
    def process(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        处理字符验证码
        
        Args:
            image: 输入图像
            **kwargs: 可能包含 ocr_engine 参数
            
        Returns:
            处理结果
        """
        try:
            ocr_engine = kwargs.get('ocr_engine', 'tesseract')
            
            # 预处理
            processed_image = self.preprocess_image(image)
            
            # OCR识别
            if ocr_engine == 'tesseract':
                result = self.ocr_with_tesseract(processed_image)
            elif ocr_engine == 'easyocr':
                result = self.ocr_with_easyocr(processed_image)
            else:
                # 尝试多种引擎
                result = self.ocr_multi_engine(processed_image)
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"字符验证码处理失败: {str(e)}")
            return self.postprocess_result({
                'success': False,
                'error': str(e)
            })
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        字符验证码预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 调整大小
        resize_height = config.get('preprocessing.char_captcha.resize_height', 60)
        if image.shape[0] != resize_height:
            image = ImageUtils.resize_image(image, height=resize_height)
        
        # 转换为灰度图
        gray = ImageUtils.to_grayscale(image)
        
        # 高斯模糊去噪
        blur_kernel = config.get('preprocessing.char_captcha.gaussian_blur_kernel', (3, 3))
        if blur_kernel[0] > 1:
            gray = ImageUtils.apply_gaussian_blur(gray, blur_kernel)
        
        # 二值化
        binary = ImageUtils.apply_threshold(gray, threshold=0, 
                                          threshold_type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作去噪
        morph_kernel = config.get('preprocessing.char_captcha.morphology_kernel', (2, 2))
        if morph_kernel[0] > 1:
            binary = ImageUtils.apply_morphology(binary, 'close', morph_kernel)
        
        return binary
    
    def ocr_with_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        使用Tesseract进行OCR识别
        
        Args:
            image: 预处理后的图像
            
        Returns:
            识别结果
        """
        if not TESSERACT_AVAILABLE:
            return {
                'success': False,
                'error': 'Tesseract不可用'
            }
        
        try:
            # OCR识别
            text = pytesseract.image_to_string(image, 
                                             config=self.tesseract_config,
                                             lang=self.tesseract_lang)
            
            # 清理文本
            clean_text = self.clean_text(text)
            
            # 获取置信度信息
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
            except:
                avg_confidence = 0
            
            result = {
                'method': 'tesseract',
                'text': clean_text,
                'raw_text': text,
                'confidence': avg_confidence,
                'success': len(clean_text) > 0
            }
            
            self.logger.info(f"Tesseract识别结果: '{clean_text}', 置信度: {avg_confidence:.1f}")
            return result
            
        except Exception as e:
            return {
                'method': 'tesseract',
                'success': False,
                'error': str(e)
            }
    
    def ocr_with_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        使用EasyOCR进行OCR识别
        
        Args:
            image: 预处理后的图像
            
        Returns:
            识别结果
        """
        if not self.easyocr_reader:
            return {
                'success': False,
                'error': 'EasyOCR不可用'
            }
        
        try:
            # OCR识别
            results = self.easyocr_reader.readtext(image, detail=1)
            
            if not results:
                return {
                    'method': 'easyocr',
                    'text': '',
                    'confidence': 0,
                    'success': False
                }
            
            # 提取文本和置信度
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                texts.append(text)
                confidences.append(conf)
            
            combined_text = ''.join(texts)
            clean_text = self.clean_text(combined_text)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            result = {
                'method': 'easyocr',
                'text': clean_text,
                'raw_text': combined_text,
                'confidence': avg_confidence * 100,  # 转换为百分比
                'details': results,
                'success': len(clean_text) > 0
            }
            
            self.logger.info(f"EasyOCR识别结果: '{clean_text}', 置信度: {avg_confidence*100:.1f}")
            return result
            
        except Exception as e:
            return {
                'method': 'easyocr',
                'success': False,
                'error': str(e)
            }
    
    def ocr_multi_engine(self, image: np.ndarray) -> Dict[str, Any]:
        """
        使用多种OCR引擎识别
        
        Args:
            image: 预处理后的图像
            
        Returns:
            识别结果
        """
        results = []
        
        # 尝试Tesseract
        if TESSERACT_AVAILABLE:
            tesseract_result = self.ocr_with_tesseract(image)
            if tesseract_result['success']:
                results.append(tesseract_result)
        
        # 尝试EasyOCR
        if self.easyocr_reader:
            easyocr_result = self.ocr_with_easyocr(image)
            if easyocr_result['success']:
                results.append(easyocr_result)
        
        if not results:
            return {
                'method': 'multi_engine',
                'success': False,
                'error': '所有OCR引擎都失败'
            }
        
        # 选择置信度最高的结果
        best_result = max(results, key=lambda x: x.get('confidence', 0))
        best_result['method'] = 'multi_engine'
        best_result['all_results'] = results
        
        return best_result
    
    def clean_text(self, text: str) -> str:
        """
        清理OCR识别的文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除空白字符
        cleaned = text.strip()
        
        # 移除换行符
        cleaned = cleaned.replace('\n', '').replace('\r', '')
        
        # 移除空格
        cleaned = cleaned.replace(' ', '')
        
        # 只保留字母数字
        cleaned = ''.join(char for char in cleaned if char.isalnum())
        
        return cleaned
    
    def segment_characters(self, image: np.ndarray) -> list:
        """
        分割字符
        
        Args:
            image: 二值化图像
            
        Returns:
            字符图像列表
        """
        # 查找轮廓
        contours = ImageUtils.find_contours(image)
        
        if not contours:
            return []
        
        # 按x坐标排序轮廓
        char_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤太小的轮廓
            if w > 5 and h > 10:
                char_boxes.append((x, y, w, h))
        
        char_boxes.sort(key=lambda x: x[0])  # 按x坐标排序
        
        # 提取字符图像
        char_images = []
        for x, y, w, h in char_boxes:
            char_img = image[y:y+h, x:x+w]
            char_images.append(char_img)
        
        return char_images 