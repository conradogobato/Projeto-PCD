%%writefile kmeans_1d_cuda.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

#define THREADS_PER_BLOCK 8  // threads por bloco
#define BLOCKS_GRID 2000         // número de blocos

// Funções auxiliares para leitura/escrita de CSV
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out) {
    int R = count_rows(path);
    double *A = (double*)malloc((size_t)R * sizeof(double));
    FILE *f = fopen(path, "r");
    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(tok) A[r++] = atof(tok);
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

// kernel (1 thread por ponto)
__global__ void assignment_kernel(const double *X, const double *C, int *assign, double *err, int N, int K){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    double xi = X[i];
    int best = -1;
    double bestd = 1e300;

    for(int c=0; c<K; c++){
        double diff = xi - C[c];
        double d = diff * diff;
        if(d < bestd){ bestd = d; best = c; }
    }
    assign[i] = best;
    err[i] = bestd; // erro quadrático para SSE
}

// Atualização na cpu: média dos pontos de cada cluster
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    for(int i=0;i<N;i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0]; // cluster vazio
    }
    free(sum); free(cnt);
}

// Função principal do K-means (host + GPU)
static void kmeans_1d_cuda(const double *X_host, double *C_host, int *assign_host,
                           int N, int K, int max_iter, double eps,
                           int *iters_out, double *sse_out)
{
    double *X_dev, *C_dev, *err_dev;
    int *assign_dev;

    size_t sizeX = N * sizeof(double);
    size_t sizeC = K * sizeof(double);
    size_t sizeA = N * sizeof(int);

    cudaMalloc(&X_dev, sizeX);
    cudaMalloc(&C_dev, sizeC);
    cudaMalloc(&assign_dev, sizeA);
    cudaMalloc(&err_dev, sizeX);
    cudaMemcpy(X_dev, X_host, sizeX, cudaMemcpyHostToDevice);

    double prev_sse = 1e300, sse = 0.0;
    int it;

    for(it=0; it<max_iter; it++){
        cudaMemcpy(C_dev, C_host, sizeC, cudaMemcpyHostToDevice);

        // Medição do tempo do kernel (GPU)
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int threads = THREADS_PER_BLOCK;
        int blocks  = BLOCKS_GRID;

        assignment_kernel<<<blocks, threads>>>(X_dev, C_dev, assign_dev, err_dev, N, K);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Fim da medição do kernel
        cudaDeviceSynchronize();

        // Copiar resultados para CPU
        cudaMemcpy(assign_host, assign_dev, sizeA, cudaMemcpyDeviceToHost);
        double *err_host = (double*)malloc(sizeX);
        cudaMemcpy(err_host, err_dev, sizeX, cudaMemcpyDeviceToHost);

        // Calcular SSE
        sse = 0.0;
        for(int i=0;i<N;i++) sse += err_host[i];
        free(err_host);

        // Critério de parada
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }

        update_step_1d(X_host, C_host, assign_host, N, K);
        prev_sse = sse;

        printf("Iter %d: Tempo do kernel = %.6f ms\n", it, kernel_ms);
    }

    *iters_out = it;
    *sse_out = sse;

    cudaFree(X_dev);
    cudaFree(C_dev);
    cudaFree(assign_dev);
    cudaFree(err_dev);
}


int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4]\n", argv[0]);
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));

    double t0 = clock();
    int iters = 0; double sse = 0.0;

    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, &iters, &sse);

    double t1 = clock();
    double total_seconds = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo total: %.6f s\n", iters, sse, total_seconds);
    printf("Configuração CUDA: %d threads por bloco, %d blocos\n", THREADS_PER_BLOCK, BLOCKS_GRID);

    write_assign_csv("assign.csv", assign, N);
    write_centroids_csv("centroids.csv", C, K);

    free(assign); free(X); free(C);
    return 0;
}





