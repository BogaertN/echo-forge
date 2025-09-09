# Contributing to EchoForge

Thank you for your interest in contributing to EchoForge! This document provides guidelines and information for contributors to help maintain code quality, project consistency, and a welcoming community environment.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community and Communication](#community-and-communication)
- [Recognition](#recognition)

## Code of Conduct

EchoForge is committed to fostering an open, inclusive, and harassment-free environment. We expect all contributors to adhere to our Code of Conduct:

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- The use of sexualized language or imagery and unwelcome sexual attention
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying standards of acceptable behavior and will take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Report any violations to [conduct@echoforge.dev](mailto:conduct@echoforge.dev).

## Getting Started

### Prerequisites

Before contributing, ensure you have:
- Python 3.9+ installed
- Git for version control
- A GitHub account
- Basic familiarity with AI/LLM concepts
- Understanding of async Python programming

### Areas for Contribution

We welcome contributions in several areas:

**Core Development:**
- Agent improvements and new agent types
- Performance optimizations
- Bug fixes and stability improvements
- Database and storage enhancements

**Frontend and UI:**
- User interface improvements
- Accessibility enhancements
- Mobile responsiveness
- Visualization features

**Testing and Quality:**
- Test coverage expansion
- Performance benchmarks
- Security audits
- Integration tests

**Documentation:**
- API documentation
- User guides and tutorials
- Code examples
- Installation guides

**Research and Innovation:**
- New debate methodologies
- AI reasoning improvements
- Knowledge graph enhancements
- Privacy and security features

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/echoforge.git
cd echoforge

# Add the upstream repository
git remote add upstream https://github.com/ORIGINAL_OWNER/echoforge.git
```

### 2. Set Up Development Environment

```bash
# Install in development mode
python install.py --dev --verbose

# Or manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Development Tools

```bash
# Code formatting and linting
pip install black isort flake8 mypy pre-commit

# Testing tools
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Install pre-commit hooks
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Start the development server
python main.py

# Check code formatting
black --check .
isort --check-only .
flake8 .
```

## Contributing Process

### 1. Choose an Issue

- Browse [open issues](https://github.com/OWNER/echoforge/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to express interest and get it assigned
- For new features, create an issue first to discuss the approach

### 2. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Changes

- Write clean, well-documented code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Commit frequently with descriptive messages

### 4. Test Your Changes

```bash
# Run the full test suite
pytest tests/ -v --cov=.

# Run linting
black .
isort .
flake8 .
mypy .

# Test manually
python main.py
```

### 5. Submit Pull Request

- Push your branch to your fork
- Create a pull request against the main branch
- Fill out the pull request template completely
- Link to any related issues

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some specific conventions:

**Formatting:**
- Use Black for code formatting (line length: 88 characters)
- Use isort for import sorting
- Use type hints for all function parameters and return values
- Use docstrings for all public functions, classes, and modules

**Example:**
```python
async def process_debate_response(
    session_id: str, 
    user_input: str, 
    context: Optional[Dict[str, Any]] = None
) -> AgentResponse:
    """
    Process user response during debate session.
    
    Args:
        session_id: Unique identifier for the debate session
        user_input: The user's response text
        context: Optional context information
        
    Returns:
        AgentResponse containing the processed result
        
    Raises:
        SessionNotFoundError: If session doesn't exist
        ValidationError: If input is invalid
    """
    if not session_id or not user_input.strip():
        raise ValidationError("Session ID and user input are required")
    
    # Implementation here...
    return AgentResponse(success=True, content="Response processed")
```

### Naming Conventions

- **Variables and functions:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private methods:** `_leading_underscore`
- **Files and modules:** `snake_case.py`

### Code Organization

```python
# Standard library imports
import asyncio
import logging
from typing import Dict, List, Optional

# Third-party imports
import fastapi
import sqlalchemy

# Local imports
from .models import DebateSession
from .utils import extract_key_concepts
```

### Error Handling

```python
# Use specific exception types
try:
    result = await process_request(data)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    raise
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    # Handle gracefully
    return default_response()
except Exception as e:
    logger.exception("Unexpected error occurred")
    raise ProcessingError("Failed to process request") from e
```

### Async/Await Guidelines

```python
# Prefer async/await over callbacks
async def fetch_data(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Use proper context managers
async def database_operation():
    async with get_db_session() as session:
        # Database operations here
        await session.commit()
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, Mock

class TestDebateOrchestrator:
    """Test suite for DebateOrchestrator class."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance for testing."""
        return DebateOrchestrator(config=test_config)
    
    @pytest.mark.asyncio
    async def test_session_creation(self, orchestrator):
        """Test creating a new debate session."""
        session_id = await orchestrator.create_session("Test question")
        assert session_id is not None
        assert len(session_id) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test proper error handling."""
        with pytest.raises(ValidationError):
            await orchestrator.create_session("")
```

### Test Categories

**Unit Tests:**
- Test individual functions and methods
- Use mocks for external dependencies
- Fast execution (< 1 second per test)

**Integration Tests:**
- Test component interactions
- Use real database (test instance)
- Medium execution time (< 10 seconds per test)

**End-to-End Tests:**
- Test complete user workflows
- Use real services when possible
- Longer execution time (< 60 seconds per test)

### Coverage Requirements

- Maintain minimum 80% test coverage
- New code should have 90%+ coverage
- Critical paths require 100% coverage

```bash
# Check coverage
pytest --cov=. --cov-report=html
# View report at htmlcov/index.html
```

## Documentation

### Code Documentation

**Docstring Format (Google Style):**
```python
def complex_function(param1: str, param2: int = 10) -> List[str]:
    """
    Brief description of what the function does.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter with default value
        
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
        
    Example:
        >>> result = complex_function("test", 5)
        >>> print(result)
        ['processed_test']
    """
```

### API Documentation

- Document all public APIs
- Include request/response examples
- Specify error conditions
- Use OpenAPI/Swagger annotations

### User Documentation

- Write clear, step-by-step instructions
- Include screenshots for UI features
- Provide code examples
- Test documentation with new users

## Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Changelog entry added (if applicable)
- [ ] No merge conflicts with main branch

### Pull Request Template

When creating a pull request, include:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Related Issues
Closes #[issue_number]

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Additional Notes
[Any additional information, considerations, or questions]
```

### Review Process

1. **Automated Checks:** CI/CD pipeline runs automatically
2. **Code Review:** At least one maintainer reviews the code
3. **Testing:** Reviewer tests the changes locally
4. **Approval:** Changes are approved and merged

### Review Criteria

**Code Quality:**
- Follows coding standards
- Proper error handling
- Efficient algorithms
- Clear variable names

**Functionality:**
- Solves the intended problem
- Doesn't break existing features
- Handles edge cases
- User-friendly error messages

**Testing:**
- Adequate test coverage
- Tests are meaningful and thorough
- Tests pass consistently

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment:** OS, Python version, EchoForge version
- **Steps to reproduce:** Detailed steps to trigger the bug
- **Expected behavior:** What should happen
- **Actual behavior:** What actually happens
- **Logs:** Relevant error messages or logs
- **Screenshots:** If applicable

### Feature Requests

Use the feature request template and include:

- **Problem description:** What problem does this solve?
- **Proposed solution:** How should it work?
- **Alternatives considered:** Other approaches you've thought about
- **Additional context:** Any other relevant information

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

## Community and Communication

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests, technical discussion
- **GitHub Discussions:** General questions, ideas, community chat
- **Discord/Slack:** Real-time chat (if available)
- **Email:** [contributors@echoforge.dev](mailto:contributors@echoforge.dev)

### Best Practices

**Be Respectful:**
- Assume good intentions
- Provide constructive feedback
- Be patient with new contributors

**Be Clear:**
- Use descriptive titles for issues and PRs
- Explain your reasoning
- Ask questions when something is unclear

**Be Collaborative:**
- Credit others for their contributions
- Share knowledge and resources
- Help review others' contributions

## Recognition

### Contributors

All contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributor graphs
- Special recognition for major contributions

### Types of Contributions

We recognize various types of contributions:

- **Code:** Bug fixes, features, performance improvements
- **Documentation:** Writing, editing, translation
- **Testing:** Writing tests, manual testing, bug reports
- **Design:** UI/UX design, graphics, user research
- **Community:** Helping others, organizing events, advocacy
- **Infrastructure:** CI/CD, deployment, monitoring

### Contribution Levels

**First-time Contributors:**
- Listed in CONTRIBUTORS.md
- Special mention in release notes

**Regular Contributors:**
- Invited to contributor team
- Access to contributor-only discussions
- Input on project roadmap

**Core Contributors:**
- Commit access (with approval process)
- Involvement in technical decisions
- Mentorship opportunities

## Development Workflow

### Branch Strategy

```
main         : Production-ready code
develop      : Integration branch for features
feature/*    : Feature development
fix/*        : Bug fixes
release/*    : Release preparation
hotfix/*     : Emergency fixes
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```
feat(agents): add specialist consultation feature

Add ability to request domain-specific expert input during debates.
Includes science, ethics, economics, history, and legal specialists.

Closes #123

fix(database): handle connection timeout gracefully

Previously, database timeouts would crash the application.
Now they are caught and handled with proper error messages.

docs(api): update websocket message format documentation

Clarify the structure of websocket messages and add examples
for all message types used in the debate flow.
```

### Release Process

1. **Feature Freeze:** Stop adding new features
2. **Testing:** Comprehensive testing and bug fixes
3. **Documentation:** Update docs and changelog
4. **Release Candidate:** Create RC for testing
5. **Final Release:** Tag and publish release
6. **Post-Release:** Monitor for issues and hotfixes

## Getting Help

### For New Contributors

- Start with issues labeled `good first issue`
- Read the codebase to understand the architecture
- Ask questions in GitHub Discussions
- Join contributor onboarding sessions (if available)

### For Experienced Contributors

- Look at `help wanted` issues
- Propose new features or improvements
- Help review pull requests
- Mentor new contributors

### Resources

- [EchoForge Documentation](https://echoforge.readthedocs.io)
- [API Reference](https://echoforge.github.io/api-docs)
- [Architecture Overview](docs/architecture.md)
- [Development FAQ](docs/development-faq.md)

## Thank You!

Thank you for contributing to EchoForge! Your efforts help make this platform better for everyone interested in exploring ideas through structured debate and reflection.

Every contribution, no matter how small, makes a difference. Whether you're fixing a typo, reporting a bug, or implementing a major feature, you're helping to build something valuable for the community.

---

*This document is a living guide and will be updated as the project evolves. If you have suggestions for improving these contribution guidelines, please open an issue or submit a pull request.*

**Happy contributing! 🚀**
