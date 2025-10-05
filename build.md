# Neural Networks LaTeX Document Build System

This directory contains a complete LaTeX document build system for your Neural Networks book/paper.

## Files Structure

```
/home/nithin/work/neural-networks/
├── main.tex           # Main LaTeX file (document root)
├── chapter1.tex       # Your chapter content
├── Makefile          # Make-based build system
├── build.sh          # Simple bash build script
├── neural_net.md     # Markdown version
└── README.md         # This file
```

## Building the Document

You have several options to build your LaTeX document:

### Option 1: Using the Build Script (Recommended for beginners)

```bash
# Compile the document
./build.sh

# Clean auxiliary files
./build.sh clean

# View the generated PDF
./build.sh view

# Get help
./build.sh help
```

### Option 2: Using Make (Recommended for advanced users)

```bash
# Compile the document
make

# Quick compile (single pass)
make quick

# Clean auxiliary files
make clean

# Clean everything including PDF
make distclean

# View PDF
make view

# Continuous compilation (requires latexmk)
make continuous

# Get help
make help
```

### Option 3: Manual Compilation

```bash
# Basic compilation
pdflatex main.tex
pdflatex main.tex  # Run twice for cross-references

# With bibliography (if you add references)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Document Features

The `main.tex` file includes:

- **Complete document structure** with book class
- **Essential packages** for math, graphics, code listings
- **Professional formatting** with proper margins and spacing
- **Syntax highlighting** for code blocks
- **Cross-references** and hyperlinks
- **Theorem environments** for definitions, examples, etc.
- **Custom commands** for mathematical notation
- **Professional title page** template

## Customization

### Adding Your Information

Edit `main.tex` and update:

- Title of the document
- Author name and affiliation
- University/institution name

### Adding More Chapters

1. Create new `.tex` files (e.g., `chapter2.tex`, `chapter3.tex`)
2. Add `\input{chapter2}` commands in `main.tex`
3. Update the Makefile dependencies if needed

### Adding Bibliography

1. Create a `references.bib` file
2. Uncomment the bibliography lines in `main.tex`:

   ```latex
   \bibliography{references}
   ```

3. Use `\cite{key}` commands in your text

### Adding Figures

Place images in a `figures/` directory and reference them:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/your-image.png}
    \caption{Your caption here}
    \label{fig:your-label}
\end{figure}
```

## Requirements

- **LaTeX distribution** (TeX Live, MiKTeX, or MacTeX)
- **pdflatex** compiler
- **Make** (for using Makefile)
- **Bash** (for using build.sh script)

### Installing LaTeX

**Ubuntu/Debian:**

```bash
sudo apt install texlive-full
```

**CentOS/RHEL:**

```bash
sudo yum install texlive-scheme-full
```

**macOS:**

```bash
brew install mactex
```

**Windows:**
Download and install MiKTeX from <https://miktex.org/>

## Troubleshooting

### Common Issues

1. **Missing packages**: Install the full LaTeX distribution
2. **Permission errors**: Make sure `build.sh` is executable (`chmod +x build.sh`)
3. **Compilation errors**: Check the `.log` file for detailed error messages

### Getting Help

- Check LaTeX error messages in the `.log` file
- Use `make help` or `./build.sh help` for available commands
- Ensure all required packages are installed

## Output

After successful compilation, you'll get:

- `main.pdf` - The final document
- Various auxiliary files (`.aux`, `.log`, `.toc`, etc.)

Use `make clean` or `./build.sh clean` to remove auxiliary files when needed.

## Next Steps

1. Customize the title page in `main.tex`
2. Add more chapters by creating additional `.tex` files
3. Include figures and references as needed
4. Consider adding an index or glossary for larger documents

Happy writing! 📝
