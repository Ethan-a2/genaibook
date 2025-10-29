# 生成式AI学习项目 - AGENTS.md

## 项目概述

这是一个与《Hands-On Generative AI with Transformers and Diffusion Models》书籍配套的代码仓库，包含了生成式AI学习的完整教程和实践代码。项目主要涵盖Transformer架构、扩散模型、Stable Diffusion、模型微调以及创意应用等核心主题。

### 主要技术栈
- **深度学习框架**: PyTorch, TensorFlow
- **核心库**: Transformers, Diffusers, Datasets
- **辅助工具**: NumPy, Matplotlib, genaibook (自定义工具包)
- **开发环境**: Jupyter Notebook, Google Colab

### 项目结构
```
genaibook/
├── 01_introduction.ipynb           # 生成式媒体介绍
├── 02_transformers.ipynb           # Transformer模型详解
├── 03_compressing.ipynb            # 信息压缩与表示
├── 04_diffusion.ipynb              # 扩散模型基础
├── 05_stable_diffusion.ipynb       # Stable Diffusion与条件生成
├── 06_fine_tuning_language_models.ipynb  # 语言模型微调
├── 07_fine_tuning_diffusion.ipynb  # 扩散模型微调
├── 08_creative_applications_of_t2i.ipynb  # 文本到图像创意应用
├── 09_generating_audio.ipynb       # 音频生成
├── 13_rag.ipynb                    # 检索增强生成(RAG)
├── pyproject.toml                  # 项目配置文件
└── README.md                       # 项目说明文档
```

## 环境配置与依赖安装

### 核心依赖安装
```bash
# 安装项目专用工具包（包含所有必需依赖）
pip install genaibook
```

### 额外依赖（特定章节需要）
```bash
# RAG章节需要
pip install langchain_community pypdf langchain-text-splitters

# 数据集处理
pip install -U datasets

# 模型加速（微调章节）
pip install accelerate
```

### 开发环境要求
- **Python**: 3.8+
- **GPU**: 推荐使用GPU环境，特别是涉及训练的章节
- **内存**: 建议8GB+ RAM
- **存储**: 部分模型需要较大存储空间

## 运行指南

### 本地运行
1. 克隆仓库到本地
2. 安装依赖：`pip install genaibook`
3. 启动Jupyter：`jupyter notebook`
4. 按章节顺序学习notebook

### Google Colab运行
1. 打开任意notebook文件
2. 使用"Open in Colab"按钮
3. 选择GPU运行时（推荐T4或更高）
4. 挂载Google Drive（如需保存进度）

### 推荐学习路径
1. **入门**: 01_introduction.ipynb → 02_transformers.ipynb
2. **基础**: 03_compressing.ipynb → 04_diffusion.ipynb
3. **进阶**: 05_stable_diffusion.ipynb → 07_fine_tuning_diffusion.ipynb
4. **应用**: 08_creative_applications_of_t2i.ipynb → 09_generating_audio.ipynb
5. **综合**: 13_rag.ipynb

## 开发规范与最佳实践

### 代码风格
- 遵循Black代码格式化（行长度80字符）
- 使用isort进行导入排序
- Jupyter notebook中可使用`skip-isort`标签跳过特定cell的导入排序

### Notebook使用规范
- 每个notebook包含书籍原代码、额外示例和练习解答
- 代码cell按逻辑顺序组织，避免重复导入
- 使用`%ls`等magic命令进行文件系统操作
- 重要的模型加载和训练步骤包含详细注释

### 性能优化建议
- 使用GPU加速：`.to(get_device())`
- 模型加载时指定精度：`torch_dtype=torch.float16`
- 批处理操作以提高效率
- 及时释放不需要的变量内存

## 常见问题与解决方案

### 内存不足
- 减少batch size
- 使用模型量化：`variant="fp16"`
- 定期清理GPU缓存：`torch.cuda.empty_cache()`

### 模型下载慢
- 使用Hugging Face镜像源
- 预下载模型到本地
- 使用`cache_dir`指定缓存路径

### 依赖冲突
- 建议使用虚拟环境：`python -m venv genaibook_env`
- 按章节安装特定依赖，避免全局污染

## 项目特色

### 教学设计
- **渐进式学习**: 从基础概念到高级应用
- **实践导向**: 每个概念都配有可运行代码
- **案例丰富**: 涵盖文本、图像、音频多模态生成

### 技术覆盖
- **Transformer架构**: GPT、BERT等模型原理与应用
- **扩散模型**: DDPM、Stable Diffusion等前沿技术
- **微调技术**: LoRA、DreamBooth等高效微调方法
- **创意应用**: ControlNet、InstructPix2Pix等实用工具

### 实用工具
- **genaibook包**: 提供设备检测、辅助函数等工具
- **预配置环境**: 开箱即用的依赖管理
- **Colab兼容**: 支持云端一键运行

## 贡献指南

### 代码贡献
- Fork项目并创建feature分支
- 确保代码通过现有测试
- 遵循项目代码风格规范
- 提交前运行格式化工具

### 问题反馈
- 在GitHub Issues中报告bug
- 提供详细的错误信息和复现步骤
- 包含系统环境信息

### 内容改进
- 欢迎提出教学建议
- 补充新的应用案例
- 优化代码注释和文档

## 联系方式

- **书籍官网**: [Hands-On Generative AI](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
- **代码仓库**: [genaibook GitHub](https://github.com/genaibook/genaibook)
- **作者团队**: Omar Sanseviero, Pedro Cuenca, Apolinario Passos, Jonathan Whitaker

---

*本文档为Mi Code AI助手生成，用于提供项目上下文和开发指导。*