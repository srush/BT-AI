.SUFFIXES: .py .notebook.py .notebook.ipynb

.notebook.py.notebook.ipynb:
	jupytext --execute --to notebook $<

.py.notebook.py:
	python -m streambook --nowatch $<


all: module1.notebook.ipynb module2.notebook.ipynb module3.notebook.ipynb module4.notebook.ipynb module5.notebook.ipynb module6.notebook.ipynb module7.notebook.ipynb module8.notebook.ipynb

clean: 
	rm *.notebook.ipynb
