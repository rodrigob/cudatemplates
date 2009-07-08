Symptom:

templates in files included from a system include path are not processed correctly


Steps to reproduce the bug:

1.) confirm correctness of source code by running "nvcc -c nvcctest1.cu"
2.) move "nvcctest2.cu" to "/usr/include" or "/usr/local/include"
3.) running "nvcc -c nvcctest1.cu" again gives the following error message:

/usr/include/nvcctest2.cu: In function ‘void launch()’:
/usr/include/nvcctest2.cu:8: error: ‘kernel’ was not declared in this scope
/usr/include/nvcctest2.cu:8: error: expected primary-expression before ‘<’ token
/usr/include/nvcctest2.cu:8: error: expected primary-expression before ‘>’ token
/usr/include/nvcctest2.cu:8: error: expected primary-expression before ‘)’ token
