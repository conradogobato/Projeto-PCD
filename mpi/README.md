## Run the code
mpirun -np [n_processos] ./kmeans_1d_mpi ../dados_random.csv ../centroids_random.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

## Rodar distribuido
mpirun -np [n_processos] -h hosts.txt\                                
  ./kmeans_1d_mpi \
  ./dados_random.csv \
  ./centroids_random.csv \
  [max_iter=50] [eps=1e-4]

