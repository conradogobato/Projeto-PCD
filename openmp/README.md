## Run on MacOS
### 1.
/opt/homebrew/bin/gcc-14 -O2 -std=c99 -f openmp openmp.cpp -o openmp -lm
### 2.
./openmp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

## Run on Ubuntu
### 1.
gcc -O2 -std=c99 -f openmp openmp.cpp -o openmp -lm
### 2.
./openmp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

### Run on Windows
### 1.
gcc -O2 -std=c99 -f openmp openmp.cpp -o openmp.exe -lm
### 2.
./openmp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]


