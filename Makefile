# Makefile for LaTeX document compilation
# Usage: make, make clean, make view, make help

# Main document name (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex

# Bibliography processor
BIBTEX = bibtex

# LaTeX source directory
LATEX_DIR = latex

# PDF viewer (adjust according to your system)
VIEWER = evince  # For Linux
# VIEWER = open    # For macOS
# VIEWER = start   # For Windows

# Default target
all: $(MAIN).pdf

# Compile the main document
$(MAIN).pdf: $(LATEX_DIR)/$(MAIN).tex $(LATEX_DIR)/chapter1.tex
	cd $(LATEX_DIR) && $(LATEX) $(MAIN).tex
	# Uncomment the next two lines if you have bibliography
	# cd $(LATEX_DIR) && $(BIBTEX) $(MAIN)
	# cd $(LATEX_DIR) && $(LATEX) $(MAIN).tex
	cd $(LATEX_DIR) && $(LATEX) $(MAIN).tex
	cp $(LATEX_DIR)/$(MAIN).pdf ./

# Quick compile (single pass)
quick: $(LATEX_DIR)/$(MAIN).tex
	cd $(LATEX_DIR) && $(LATEX) $(MAIN).tex
	cp $(LATEX_DIR)/$(MAIN).pdf ./

# Clean auxiliary files
clean:
	cd $(LATEX_DIR) && rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.lof *.lot *.nav *.snm *.vrb *.synctex.gz
	rm -f $(MAIN).pdf

# Clean everything including PDF
distclean: clean
	cd $(LATEX_DIR) && rm -f $(MAIN).pdf

# View the generated PDF
view: $(MAIN).pdf
	$(VIEWER) $(MAIN).pdf &

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Compile the complete document (default)"
	@echo "  quick     - Quick compile (single LaTeX pass)"
	@echo "  clean     - Remove auxiliary files"
	@echo "  distclean - Remove all generated files including PDF"
	@echo "  view      - Open the PDF with default viewer"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Note: LaTeX source files are in the '$(LATEX_DIR)/' directory"
	@echo "      Generated PDF will be copied to the root directory"

# Force rebuild
force:
	$(MAKE) clean
	$(MAKE) all

# Continuous compilation (requires latexmk)
continuous:
	cd $(LATEX_DIR) && latexmk -pdf -pvc $(MAIN).tex

.PHONY: all quick clean distclean view help force continuous