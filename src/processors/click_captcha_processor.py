"""
点击验证码处理器
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from .base_processor import BaseProcessor
from ..utils import ImageUtils, config
import cv2
import os

class ClickCaptchaProcessor(BaseProcessor):
    """点击验证码处理器"""
    
    def __init__(self):
        super().__init__()
        self.confidence_threshold = config.get('yolo.confidence', 0.5)
        
    def process(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        处理点击验证码
        
        Args:
            image: 输入图像
            **kwargs: 可能包含 target_description, templates, template_paths
            
        Returns:
            处理结果
        """
        try:
            # 检查是否有模板图像用于匹配
            templates = kwargs.get('templates', [])
            template_paths = kwargs.get('template_paths', [])
            target_description = kwargs.get('target_description', '')
            
            if templates or template_paths:
                # 使用模板匹配方法
                result = self.process_template_matching(image, templates, template_paths)
            else:
                # 使用简单的轮廓检测方法
                result = self.detect_objects_simple(image, target_description)
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"点击验证码处理失败: {str(e)}")
            return self.postprocess_result({
                'success': False,
                'error': str(e)
            })
    
    def process_template_matching(self, main_image: np.ndarray, 
                                templates: List[np.ndarray] = None,
                                template_paths: List[str] = None) -> Dict[str, Any]:
        """
        使用模板匹配处理点击验证码（如极验4代）
        
        Args:
            main_image: 主图像
            templates: 模板图像列表
            template_paths: 模板图像路径列表
            
        Returns:
            检测结果
        """
        self.logger.info("开始模板匹配检测")
        
        # 加载模板图像
        if template_paths and not templates:
            templates = []
            template_names = []
            for path in template_paths:
                if os.path.exists(path):
                    template = cv2.imread(path)
                    if template is not None:
                        templates.append(template)
                        template_names.append(os.path.basename(path))
                        self.logger.info(f"加载模板: {path}")
                    else:
                        self.logger.warning(f"无法加载模板: {path}")
                else:
                    self.logger.warning(f"模板文件不存在: {path}")
        else:
            template_names = [f"template_{i}" for i in range(len(templates))] if templates else []
        
        if not templates:
            return {
                'success': False,
                'error': '没有可用的模板图像',
                'targets': [],
                'target_count': 0
            }
        
        targets = []
        
        # 对每个模板进行匹配
        for i, template in enumerate(templates):
            template_name = template_names[i] if i < len(template_names) else f"template_{i}"
            matches = self.match_template(main_image, template, template_name)
            targets.extend(matches)
        
        # 按置信度排序
        targets.sort(key=lambda x: x['confidence'], reverse=True)
        
        result = {
            'method': 'template_matching',
            'targets': targets,
            'target_count': len(targets),
            'success': len(targets) > 0,
            'template_count': len(templates)
        }
        
        self.logger.info(f"模板匹配完成，找到 {len(targets)} 个目标")
        return result
    
    def match_template(self, main_image: np.ndarray, template: np.ndarray, 
                      template_name: str = "unknown") -> List[Dict[str, Any]]:
        """
        单个模板匹配
        
        Args:
            main_image: 主图像
            template: 模板图像
            template_name: 模板名称
            
        Returns:
            匹配结果列表
        """
        # 获取模板尺寸
        template_height, template_width = template.shape[:2]
        
        # 转换为灰度图像进行匹配
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY) if len(main_image.shape) == 3 else main_image
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        # 模板匹配
        result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # 设置匹配阈值
        threshold = 0.6  # 可以根据需要调整
        locations = np.where(result >= threshold)
        
        matches = []
        
        # 获取所有高于阈值的匹配位置
        for pt in zip(*locations[::-1]):  # Switch columns and rows
            x, y = pt
            center_x = x + template_width // 2
            center_y = y + template_height // 2
            confidence = result[y, x]
            
            match_info = {
                'template_name': template_name,
                'center': (center_x, center_y),
                'top_left': (x, y),
                'bottom_right': (x + template_width, y + template_height),
                'bbox': (x, y, template_width, template_height),
                'confidence': float(confidence),
                'template_size': (template_width, template_height)
            }
            matches.append(match_info)
        
        # 去除重复的匹配
        matches = self.remove_duplicate_matches(matches)
        
        self.logger.info(f"模板 {template_name} 找到 {len(matches)} 个匹配")
        for match in matches:
            self.logger.info(f"  - 位置: {match['center']}, 置信度: {match['confidence']:.3f}")
        
        return matches
    
    def remove_duplicate_matches(self, matches: List[Dict[str, Any]], 
                                min_distance: int = 30) -> List[Dict[str, Any]]:
        """
        去除距离过近的重复匹配
        
        Args:
            matches: 匹配结果列表
            min_distance: 最小距离阈值
            
        Returns:
            去重后的匹配列表
        """
        if len(matches) <= 1:
            return matches
        
        # 按置信度排序
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_matches = []
        
        for match in matches:
            is_duplicate = False
            center = match['center']
            
            for existing_match in filtered_matches:
                existing_center = existing_match['center']
                distance = np.sqrt((center[0] - existing_center[0])**2 + 
                                 (center[1] - existing_center[1])**2)
                
                if distance < min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def detect_objects_simple(self, image: np.ndarray, 
                             target_description: str) -> Dict[str, Any]:
        """
        简单的目标检测方法（基于轮廓）
        
        Args:
            image: 输入图像
            target_description: 目标描述
            
        Returns:
            检测结果
        """
        self.logger.info(f"开始检测目标: {target_description}")
        
        # 预处理
        gray = ImageUtils.to_grayscale(image)
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours = ImageUtils.find_contours(edges)
        
        # 分析轮廓，找到可能的目标
        targets = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤太小或太大的轮廓
            if 100 < area < 10000:
                center_x = x + w // 2
                center_y = y + h // 2
                
                targets.append({
                    'id': i,
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': 0.8  # 简单的固定置信度
                })
        
        result = {
            'method': 'simple_contour_detection',
            'target_description': target_description,
            'targets': targets,
            'target_count': len(targets),
            'success': len(targets) > 0
        }
        
        self.logger.info(f"检测到 {len(targets)} 个目标")
        return result
    
    # TODO: 集成YOLO模型
    def detect_objects_yolo(self, image: np.ndarray, 
                           target_class: str) -> Dict[str, Any]:
        """
        使用YOLO模型检测目标（待实现）
        
        Args:
            image: 输入图像
            target_class: 目标类别
            
        Returns:
            检测结果
        """
        # 这里可以集成YOLO模型
        return {
            'method': 'yolo',
            'success': False,
            'error': 'YOLO模型未实现'
        } 