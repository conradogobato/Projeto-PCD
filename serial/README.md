## Run on MacOS
### 1.
/opt/homebrew/bin/gcc-14 -O2 -std=c99 -f n_parallel n_parallel.cpp -o n_parallel -lm
### 2.
./n_parallel dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

## Run on Ubuntu
### 1.
gcc -O2 -std=c99 -f n_parallel n_parallel.cpp -o n_parallel -lm
### 2.
./n_parallel dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

### Run on Windows
### 1.
gcc -O2 -std=c99 -f n_parallel n_parallel.cpp -o n_parallel.exe -lm
### 2.
./n_parallel dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]


