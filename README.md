# Payout Approximation Research

> ⚠️ **Note**: This research paper is currently under active development.

This repository contains source code and LaTeX files for a research paper on approximating complex payout profiles using vanilla options with regularization techniques.

## Project Structure

- `code/` - Python implementation
  - `approximation.py` - Core implementation of regularized payout approximation
- `latex/` - LaTeX source files
  - `article.tex` - Main paper source
  - `references.bib` - Bibliography
- `build/` - Generated files (PDF output)

## Features

- Implementation of L1 and L2 regularized payout approximation
- Comparison of different regularization methods
- Automatic generation of TikZ plots
- Integration of Python computations with LaTeX document

## Usage

The project uses Make for automation. Available commands:

```bash
make demo   # Run full demonstration (generate plots and compile PDF)
make all    # Build the PDF
make clean  # Clean generated files
make watch  # Watch for changes and rebuild
make help   # Show all available commands
```

## Requirements

- Python with NumPy, SciPy, Matplotlib
- LaTeX with TikZ/PGFPlots
- Make

## Author

```
@misc{
    Kochetov2025,
    author   = {Kochetov, Vlad},
    title    = {From Distribution Forecasts to Vanilla Portfolios: Regularized Payout Approximation in Discrete Space},
  year     = {2025},
}
```