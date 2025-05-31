# Image Verification - 验证码识别框架

一个基于OpenCV、OCR和YOLO的多类型验证码识别框架，支持字符验证码、滑块验证码、点击验证码和拼图验证码的自动识别。

## 🚀 特性

- **多类型支持**: 支持4种主要验证码类型
- **自动检测**: 智能识别验证码类型
- **模块化设计**: 易于扩展和定制
- **多OCR引擎**: 支持Tesseract和EasyOCR
- **配置化**: 通过YAML文件灵活配置
- **详细日志**: 完整的处理过程记录

## 📁 项目结构

```
Image-verification/
├── src/                          # 源代码
│   ├── __init__.py
│   ├── core/                     # 核心模块
│   │   ├── captcha_detector.py   # 验证码类型检测器
│   │   └── captcha_processor.py  # 主处理器
│   ├── processors/               # 各类型处理器
│   │   ├── base_processor.py     # 基础处理器
│   │   ├── char_captcha_processor.py    # 字符验证码
│   │   ├── slider_captcha_processor.py  # 滑块验证码
│   │   ├── click_captcha_processor.py   # 点击验证码
│   │   └── puzzle_captcha_processor.py  # 拼图验证码
│   └── utils/                    # 工具模块
│       ├── config.py             # 配置管理
│       ├── logger.py             # 日志管理
│       └── image_utils.py        # 图像处理工具
├── config/
│   └── config.yaml               # 配置文件
├── example/                      # 示例验证码
├── main.py                       # 主程序
├── analyze_examples.py           # 图像分析工具
├── requirements.txt              # 依赖包
└── README.md
```

## 🛠️ 安装

1. **克隆项目**
```bash
git clone <repository-url>
cd Image-verification
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **安装Tesseract OCR** (可选)
- Windows: 下载并安装 [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Ubuntu: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## 🎯 支持的验证码类型

### 1. 字符验证码 (OCR识别)
- 传统的字母数字验证码
- 支持Tesseract和EasyOCR双引擎
- 自动图像预处理和文本清理

### 2. 滑块验证码 (模板匹配)
- 检测缺口位置
- 计算滑动距离
- 生成滑动轨迹

### 3. 点击验证码 (目标检测)
- 基于轮廓的简单检测
- 预留YOLO模型接口
- 支持多目标识别

### 4. 拼图验证码 (图像差分)
- 模板匹配定位
- 形状特征分析
- 支持背景图匹配

## 📖 使用方法

### 基本使用

```python
from src import CaptchaProcessor
from src.core.captcha_detector import CaptchaType

# 初始化处理器
processor = CaptchaProcessor()

# 自动检测并处理
result = processor.process_captcha("path/to/captcha.jpg")
print(f"类型: {result['captcha_type']}")
print(f"结果: {result}")

# 指定验证码类型
result = processor.process_captcha(
    "path/to/captcha.jpg", 
    captcha_type=CaptchaType.CHAR_CAPTCHA
)
```

### 字符验证码

```python
# 使用特定OCR引擎
result = processor.process_char_captcha(image, ocr_engine='tesseract')
print(f"识别文本: {result['text']}")
print(f"置信度: {result['confidence']}")
```

### 滑块验证码

```python
# 背景图和滑块图
result = processor.process_slider_captcha(background_img, slider_img)
print(f"滑动距离: {result['distance']}px")
print(f"位置: {result['position']}")

# 只有背景图，检测缺口
result = processor.process_captcha(background_img, 
                                 captcha_type=CaptchaType.SLIDER_CAPTCHA)
```

### 点击验证码

```python
result = processor.process_click_captcha(image, 
                                       target_description="点击所有汽车")
print(f"检测到 {result['target_count']} 个目标")
for target in result['targets']:
    print(f"目标位置: {target['center']}")
```

## 🔧 配置

编辑 `config/config.yaml` 来自定义设置：

```yaml
# OCR配置
ocr:
  tesseract:
    config: '--oem 3 --psm 6'
    language: 'eng'
  easyocr:
    languages: ['en']
    gpu: false

# YOLO配置
yolo:
  confidence: 0.5
  device: 'cpu'

# 预处理配置
preprocessing:
  char_captcha:
    resize_height: 60
    gaussian_blur_kernel: (3, 3)
```

## 🚀 快速开始

1. **运行示例分析**
```bash
python analyze_examples.py
```

2. **运行主程序**
```bash
python main.py
```

3. **查看结果**
结果将保存在 `results/captcha_results.json`

## 📊 性能优化

### 字符验证码
- 调整预处理参数提高OCR准确率
- 使用多引擎结果融合
- 针对特定字体训练模型

### 滑块验证码
- 优化边缘检测参数
- 使用更精确的模板匹配算法
- 添加轨迹生成算法

### 点击验证码
- 集成YOLO目标检测模型
- 训练特定类别的检测器
- 优化后处理算法

## 🔮 扩展开发

### 添加新的验证码类型

1. 在 `CaptchaType` 枚举中添加新类型
2. 创建对应的处理器继承 `BaseProcessor`
3. 在 `CaptchaProcessor` 中注册新处理器
4. 更新检测器的分类逻辑

### 集成YOLO模型

```python
# 在 ClickCaptchaProcessor 中
def detect_objects_yolo(self, image, target_class):
    from ultralytics import YOLO
    model = YOLO('path/to/model.pt')
    results = model(image)
    # 处理结果...
```

## 🤔 常见问题

**Q: Tesseract识别准确率低怎么办？**
A: 尝试调整预处理参数，或使用EasyOCR引擎，也可以训练专用模型。

**Q: 滑块验证码检测不准确？**
A: 检查边缘检测阈值设置，确保缺口特征明显。

**Q: 如何添加新的OCR引擎？**
A: 在 `CharCaptchaProcessor` 中添加新的OCR方法，并更新配置文件。

## 📝 开发计划

- [ ] 集成YOLO模型用于点击验证码
- [ ] 添加深度学习字符识别模型
- [ ] 支持动态验证码（GIF/视频）
- [ ] 添加验证码生成器用于测试
- [ ] 优化滑动轨迹生成算法
- [ ] 添加Web API接口

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 🧩 验证码类型对比

| 验证码类型 | 常规技术路径 | 推荐方法 | 适用场景 |
|------------|----------------|-------------|----------|
| 文字验证码 | 图像处理 + OCR | Tesseract, EasyOCR | 传统网站登录 |
| 滑块验证码 | 模板匹配 | OpenCV | 现代网站防护 |
| 点击验证码 | 多目标检测 | YOLO | 高安全要求 |
| 拼图验证码 | 差分/检测 | OpenCV + YOLO | 游戏化验证 |

## ✅ 总结

这个框架提供了完整的验证码识别解决方案，支持主流的验证码类型。通过模块化设计，可以轻松扩展和定制。无论是研究学习还是实际应用，都能提供良好的基础。