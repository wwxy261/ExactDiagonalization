# 费米Hubbard模型严格对角化的算法实现

## Ground State   （done）

## Green Function  (to do)


## Python 版本
Python使用SciPy库进行Sparse Matrix的构建以及求解，并尽可能使用Numba库优化，性能瓶颈在Hamiltonian Matrix的构建(Numba 暂时不支持对SciPy的优化)

## C++ 版本
现代C++风格编写，Sparse Matrix库使用Eigen，最小特征值的求解使用Spectra/ARPACK。性能瓶颈在最小特征值的求解(Eigen 对ARPACK的支持还不完善)

### 编译
```
g++ -O3 -std=c++17 -I include/ src/ED.cpp -o ED  
```


