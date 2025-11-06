# Contributing to Universal Scraper

Thank you for your interest in contributing to Universal Scraper! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/universal-scraper.git
   cd universal-scraper
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Development Setup

### Running Tests

```bash
pytest tests/
```

### Code Style

We follow PEP 8 guidelines. Before submitting:

```bash
# Format code
black universal_scraper/

# Check linting
flake8 universal_scraper/

# Type checking
mypy universal_scraper/
```

## 📝 Making Changes

### Branch Naming

- `feature/your-feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `docs/documentation-update` - Documentation changes

### Commit Messages

Follow conventional commits format:

- `feat: add new feature`
- `fix: resolve bug`
- `docs: update documentation`
- `test: add tests`
- `refactor: refactor code`

### Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## 🧪 Testing Guidelines

- Write tests for all new features
- Maintain test coverage above 80%
- Include both unit and integration tests
- Test with multiple AI providers (OpenAI, Gemini, Claude)

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Include examples for new features
- Update API reference if needed

## 🐛 Bug Reports

When filing a bug report, include:

1. Python version
2. OS and version
3. Universal Scraper version
4. Minimal reproducible example
5. Expected vs actual behavior
6. Relevant logs/error messages

## 💡 Feature Requests

For feature requests, describe:

1. Use case and motivation
2. Proposed solution
3. Alternative solutions considered
4. Impact on existing functionality

## 🎯 Priority Areas

Current priority areas for contributions:

1. **AI Provider Support**: Add more AI providers
2. **Performance**: Optimize HTML cleaning and caching
3. **JSON Detection**: Improve JSON source detection
4. **Documentation**: More examples and tutorials
5. **Testing**: Increase test coverage

## 📞 Getting Help

- Open an issue for questions
- Join discussions in GitHub Discussions
- Check existing issues and PRs first

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards others

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Universal Scraper! 🎉

