"""
验证码处理器模块
"""

from .char_captcha_processor import CharCaptchaProcessor
from .slider_captcha_processor import SliderCaptchaProcessor
from .click_captcha_processor import ClickCaptchaProcessor
from .puzzle_captcha_processor import PuzzleCaptchaProcessor

__all__ = [
    'CharCaptchaProcessor',
    'SliderCaptchaProcessor', 
    'ClickCaptchaProcessor',
    'PuzzleCaptchaProcessor'
] 