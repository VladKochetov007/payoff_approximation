.PHONY: all clean watch help

# Default target
all: build/article.pdf

# Main PDF build
build/article.pdf: src/article.tex src/references.bib
	latexmk -pdf

# Clean all generated files
clean:
	latexmk -C
	rm -rf build/

# Watch for changes and rebuild
watch:
	latexmk -pdf -pvc

# Help target
help:
	@echo "Available targets:"
	@echo "  make        - Build the PDF (same as 'make all')"
	@echo "  make all    - Build the PDF"
	@echo "  make clean  - Remove all generated files"
	@echo "  make watch  - Watch for changes and rebuild automatically"
	@echo "  make help   - Show this help message" 