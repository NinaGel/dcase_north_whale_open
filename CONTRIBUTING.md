# Contributing Guide / 贡献指南

Thank you for your interest in contributing to this project!

感谢您有兴趣为本项目做出贡献！

## How to Report Issues / 如何报告问题

1. Check if the issue already exists in [GitHub Issues](https://github.com/NinaGel/dcase_north_whale_open/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Environment details (Python version, PyTorch version, OS)

1. 检查问题是否已存在于 [GitHub Issues](https://github.com/NinaGel/dcase_north_whale_open/issues)
2. 如果没有，创建新的 issue，包含：
   - 清晰的描述性标题
   - 复现问题的步骤
   - 预期行为与实际行为
   - 环境详情（Python版本、PyTorch版本、操作系统）

## How to Submit Code / 如何提交代码

### 1. Fork and Clone / Fork 并克隆

```bash
git clone https://github.com/YOUR_USERNAME/dcase_north_whale_open.git
cd dcase_north_whale_open
```

### 2. Create a Branch / 创建分支

```bash
git checkout -b feature/your-feature-name
# or / 或者
git checkout -b fix/your-bug-fix
```

### 3. Make Changes / 进行修改

- Follow the existing code style
- Add bilingual comments (Chinese + English) for new code
- Run tests before committing

- 遵循现有的代码风格
- 为新代码添加双语注释（中文 + 英文）
- 提交前运行测试

### 4. Commit and Push / 提交并推送

```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request / 创建 Pull Request

- Open a PR against the `main` branch
- Describe your changes clearly
- Link related issues if applicable

- 针对 `main` 分支打开 PR
- 清晰描述您的更改
- 如适用，链接相关的 issue

## Code Standards / 代码规范

### Python Style / Python 风格

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Keep functions focused and small

- 遵循 PEP 8 指南
- 使用有意义的变量和函数名
- 保持函数专注且简短

### Comments / 注释

Use bilingual comments for important code sections:

对重要的代码部分使用双语注释：

```python
def forward(self, x):
    """前向传播 / Forward propagation

    Args:
        x: 输入张量 / Input tensor [B, C, H, W]

    Returns:
        输出特征 / Output features
    """
    # 特征提取 / Feature extraction
    x = self.conv(x)
    return x
```

## Development Setup / 开发环境设置

```bash
# Create virtual environment / 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or / 或者
venv\Scripts\activate  # Windows

# Install dependencies / 安装依赖
pip install -r requirements.txt

# Install dev dependencies (optional) / 安装开发依赖（可选）
pip install pytest pytest-cov black flake8 isort
```

## Questions? / 有问题?

Feel free to open an issue or start a discussion!

欢迎开启 issue 或发起讨论！
