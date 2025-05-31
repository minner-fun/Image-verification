"""
日志管理工具
"""
import sys
from pathlib import Path
from loguru import logger
from .config import config

class LogManager:
    """日志管理器"""
    
    def __init__(self):
        self.setup_logger()
    
    def setup_logger(self):
        """设置日志配置"""
        # 移除默认的 logger
        logger.remove()
        
        # 获取日志级别
        log_level = config.get('output.log_level', 'INFO')
        
        # 控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=log_level,
            colorize=True
        )
        
        # 文件输出
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        
        logger.add(
            log_path / "image_verification_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} | {message}",
            level=log_level,
            rotation="1 day",
            retention="7 days",
            compression="zip",
            encoding="utf-8"
        )
    
    def get_logger(self, name: str = None):
        """获取logger实例"""
        if name:
            return logger.bind(name=name)
        return logger

# 全局日志实例
log_manager = LogManager()
get_logger = log_manager.get_logger 