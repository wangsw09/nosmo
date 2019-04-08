.PHONY: clean

clean:
	rm -rf alo/coptimization/*.c build alo/coptimization/*.so alo/coptimization/*~ alo/coptimization/.*~ alo/coptimization/*.pyc alo/clinalg/*.c build alo/clinalg/*.so alo/clinalg/*~ alo/clinalg/.*~ alo/clinalg/*.pyc alo/cfuncs/*.c build alo/cfuncs/*.so alo/cfuncs/*~ alo/cfuncs/.*~ alo/cfuncs/*.pyc alo/calo/*.c build alo/calo/*.so alo/calo/*~ alo/calo/.*~ alo/calo/*.pyc alo/cross_validate/*.c build alo/cross_validate/*.so alo/cross_validate/*~ alo/cross_validate/.*~ alo/cross_validate/*.pyc


compile:
	python setup.py build_ext --inplace

post_clean:
	rm -rf build

all: compile 
