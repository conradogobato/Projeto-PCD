# K-Means 1D — CUDA

## Run on Google Colab
### 1.
!nvcc -O2 -std=c++11 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda
### 2.
!./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

---

## Run on Ubuntu / Linux
### 1.
nvcc -O2 -std=c++11 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda
### 2.
./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

---

## Run on Windows (WSL2 recomendado)
### 1.
nvcc -O2 -std=c++11 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda.exe
### 2.
./kmeans_1d_cuda.exe dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
