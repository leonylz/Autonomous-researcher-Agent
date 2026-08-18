# Contributing to Deep Researcher Agent

Thanks for your interest! Here's how to contribute.

## Areas We Need Help

- **Cloud GPU Platforms**: AWS, GCP, Lambda Labs, RunPod support
- **Experiment Trackers**: W&B, MLflow, TensorBoard integration
- **Research Skills**: New Claude Code slash commands
- **Metric Extraction**: Better parsing of training logs across frameworks

## How to Contribute

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Test locally
5. Submit a PR with a clear description

## Code Style

- Python 3.10+ with type hints
- Follow existing patterns in `core/`
- Keep files under 400 lines
- Docstrings for public functions

## Reporting Issues

Please include:
- Python version
- GPU setup (nvidia-smi output)
- Steps to reproduce
- Error logs
