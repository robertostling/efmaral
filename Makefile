all: cyalign.so

clean:
	rm -f *.so cyalign.c

cyalign.so: cyalign.pyx gibbs.c intmap.c
	python3 setup.py build_ext --inplace

