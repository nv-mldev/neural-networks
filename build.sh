#!/bin/bash

# Build script for Neural Networks LaTeX document
# This script provides an easy way to compile the document

echo "=== Neural Networks Document Build Script ==="
echo

# LaTeX source directory
LATEX_DIR="latex"

# Check if required files exist
if [ ! -f "$LATEX_DIR/main.tex" ]; then
    echo "Error: $LATEX_DIR/main.tex not found!"
    exit 1
fi

if [ ! -f "$LATEX_DIR/chapter1.tex" ]; then
    echo "Error: $LATEX_DIR/chapter1.tex not found!"
    exit 1
fi

# Function to compile document
compile_document() {
    echo "Compiling document from $LATEX_DIR/ directory..."
    
    echo "First pass..."
    cd "$LATEX_DIR"
    pdflatex -interaction=nonstopmode main.tex
    
    echo "Second pass (for cross-references)..."
    pdflatex -interaction=nonstopmode main.tex
    
    if [ $? -eq 0 ]; then
        echo "✓ Compilation successful!"
        cd ..
        cp "$LATEX_DIR/main.pdf" ./
        echo "Generated: main.pdf (copied to root directory)"
    else
        echo "✗ Compilation failed!"
        echo "Check the .log file in $LATEX_DIR/ for errors."
        cd ..
        exit 1
    fi
}

# Function to clean auxiliary files
clean_files() {
    echo "Cleaning auxiliary files..."
    cd "$LATEX_DIR"
    rm -f *.aux *.log *.toc *.lof *.lot *.out *.fdb_latexmk *.fls *.synctex.gz
    cd ..
    rm -f main.pdf
    echo "✓ Cleaned!"
}

# Function to view PDF
view_pdf() {
    if [ -f "main.pdf" ]; then
        echo "Opening PDF..."
        # Detect OS and use appropriate viewer
        if command -v evince &> /dev/null; then
            evince main.pdf &
        elif command -v open &> /dev/null; then
            open main.pdf
        elif command -v xdg-open &> /dev/null; then
            xdg-open main.pdf &
        else
            echo "No PDF viewer found. Please open main.pdf manually."
        fi
    else
        echo "Error: main.pdf not found! Compile first."
    fi
}

# Parse command line arguments
case "$1" in
    "clean")
        clean_files
        ;;
    "view")
        view_pdf
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  (no args)  - Compile the document"
        echo "  clean      - Remove auxiliary files"
        echo "  view       - Open the PDF"
        echo "  help       - Show this help"
        echo
        echo "Examples:"
        echo "  $0         # Compile document"
        echo "  $0 clean   # Clean files"
        echo "  $0 view    # View PDF"
        ;;
    "")
        compile_document
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for available commands."
        exit 1
        ;;
esac