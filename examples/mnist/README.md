# HMM: Handwritten Digits Recognition

This example use datas from [MNIST](http://yann.lecun.com/exdb/mnist/).  
First thing first, please download and extract the data files - `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` - into `./datas/` directory.  


## Result
```
test 9900 datas
viterby correct rate each class:
[ 0.90092879  0.93499555  0.78787879  0.828       0.91340206  0.66252822
  0.88160677  0.83677483  0.69637306  0.8493014 ]
forward correct rate each class:
[ 0.8998968   0.95903829  0.74780059  0.807       0.91237113  0.64785553
  0.87420719  0.83087512  0.72227979  0.83133733]
viterby total correct rate: 0.82917894541
forward total correct rate: 0.823266177129
```
