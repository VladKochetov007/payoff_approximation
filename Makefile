.PHONY: all clean watch help demo

TEX_DIR = latex
BUILD_DIR = build

# Default target
all: $(BUILD_DIR)/article.pdf

# Main PDF build
$(BUILD_DIR)/article.pdf: $(TEX_DIR)/article.tex $(TEX_DIR)/references.bib
	latexmk -pdf -outdir=$(BUILD_DIR) $(TEX_DIR)/article.tex

# Clean all generated files
clean:
	latexmk -C
	rm -rf $(BUILD_DIR)/

# Watch for changes and rebuild
watch:
	latexmk -pdf -pvc -outdir=$(BUILD_DIR) $(TEX_DIR)/article.tex

# Demo target - run code, build PDF and show
demo:
	@echo "Cleaning..."
	@make clean
	@echo "Running Python code..."
	@cd code && python approximation.py
	@echo "Building PDF..."
	@make all
	@echo "Opening PDF..."
	@xdg-open $(BUILD_DIR)/article.pdf

# Help target
help:
	@echo "Available targets:"
	@echo "  make        - Build the PDF (same as 'make all')"
	@echo "  make all    - Build the PDF"
	@echo "  make clean  - Remove all generated files"
	@echo "  make watch  - Watch for changes and rebuild automatically"
	@echo "  make demo   - Run Python code, build PDF and show result"
	@echo "  make help   - Show this help message" 