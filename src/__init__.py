"""
Image Verification Package
图片验证码识别工具包

支持的验证码类型：
- 字符验证码 (OCR识别)
- 滑块验证码 (模板匹配)  
- 点击验证码 (目标检测)
- 拼图验证码 (图像差分)
"""

__version__ = "1.0.0"
__author__ = "Image Verification Team"

from .core.captcha_detector import CaptchaDetector
from .core.captcha_processor import CaptchaProcessor

__all__ = [
    'CaptchaDetector',
    'CaptchaProcessor'
] 