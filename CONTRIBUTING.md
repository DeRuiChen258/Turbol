# Contributing to TurboRL

Thank you for your interest in contributing to TurboRL!

## How to Contribute

### Reporting Issues

- Search existing issues before creating a new one
- Use issue templates when available
- Include reproduction steps and expected behavior

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow Google C++ Style Guide
- Run tests before submitting
- Add tests for new features

### Build & Test

```bash
# Build
cd turbol_vllm
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DTURBORL_ENABLE_CUDA=ON \
    -DTURBORL_ENABLE_TESTS=ON
cmake --build . -j$(nproc)

# Test
./turbol_test
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove)
- Reference issues when applicable

## Development Guidelines

### C++ Standards
- C++17 minimum
- CUDA 13.x compatible
- Cross-platform support (Linux, macOS, Windows)

### Testing
- All new features must include tests
- Maintain 100% test pass rate
- Include performance benchmarks for critical paths

### Documentation
- Update README.md for user-facing changes
- Add inline comments for complex code
- Document API changes

## Questions?

Feel free to open an issue for any questions about contributing.
