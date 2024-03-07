# Notes for the paper 
## code
All code samples should go into `/code` and can be then included in the paper as follows 
```latex
\lstinputlisting[firstline=10, lastline=13]{code/sample_script.py}
```

## images
All assets, so mainly images should go in assets can can be included as follows 
```latex
\includegraphics[width=0.5\textwidth]{sample_image} % latex knows its in assets ( see pagestyle )
```

## additional
You can use `make clean` to remove the latexmk files, they should already be hidden by the `.vscode` config