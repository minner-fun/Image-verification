"""
配置管理工具
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config/config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"配置文件 {self.config_path} 未找到，使用默认配置")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"配置文件解析错误: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'ocr': {
                'tesseract': {
                    'config': '--oem 3 --psm 6',
                    'language': 'eng'
                },
                'easyocr': {
                    'languages': ['en'],
                    'gpu': False,
                    'detail': 0
                }
            },
            'yolo': {
                'confidence': 0.5,
                'iou_threshold': 0.45,
                'device': 'cpu'
            },
            'template_matching': {
                'threshold': 0.8
            },
            'output': {
                'save_results': True,
                'result_path': 'results/',
                'log_level': 'INFO'
            }
        }
    
    def get(self, key: str, default=None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分割格式如 'ocr.tesseract.config'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """保存配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)

# 全局配置实例
config = ConfigManager() 