.PHONY: clean

clean:
	rm -rf nosmo/coptimization/*.c build nosmo/coptimization/*.so nosmo/coptimization/*~ nosmo/coptimization/.*~ nosmo/coptimization/*.pyc nosmo/clinalg/*.c build nosmo/clinalg/*.so nosmo/clinalg/*~ nosmo/clinalg/.*~ nosmo/clinalg/*.pyc nosmo/cfuncs/*.c build nosmo/cfuncs/*.so nosmo/cfuncs/*~ nosmo/cfuncs/.*~ nosmo/cfuncs/*.pyc nosmo/cnosmo/*.c build nosmo/cnosmo/*.so nosmo/cnosmo/*~ nosmo/cnosmo/.*~ nosmo/cnosmo/*.pyc nosmo/cross_validate/*.c build nosmo/cross_validate/*.so nosmo/cross_validate/*~ nosmo/cross_validate/.*~ nosmo/cross_validate/*.pyc


compile:
	python setup.py build_ext --inplace

post_clean:
	rm -rf build

all: clean compile post_clean
