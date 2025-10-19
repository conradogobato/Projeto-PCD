/* kmeans_1d_naive.cpp
   K-means 1D com OpenMP:
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).

   Compilar: gcc -O2 -std=c99 -fopenmp kmeans_1d_naive.cpp -o kmeans_1d_naive -lm
   Uso:      ./kmeans_1d_naive dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>



/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
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
/* A função abre um arquivo, conta quantas linhas contêm algum caractere válido (diferente de espaço/tabulação/quebra de linha), e retorna esse número. */

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

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
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}
/* A função cria o vetor com os valores lidos no arquivo, e armazena o numero de elementos na memória */


static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- k-means 1D com OpenMP ---------- */
static double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;

    #pragma omp parallel
    {
        double sse_local = 0.0;

        #pragma omp for  // Seção não crítica divide o array de pontos entre as threads
        for(int i=0;i<N;i++){
            int best = -1;
            double bestd = 1e300;
            for(int c=0;c<K;c++){
                double diff = X[i] - C[c];
                double d = diff*diff;
                if(d < bestd){ bestd = d; best = c; }
            }
            assign[i] = best;
            sse_local += bestd;
        }

        #pragma omp critical // Seção crítica para somar os erros locais ao erro global
        {
            sse += sse_local;
        }
    }

    return sse;
}

static void update_step_1d(const double *X, double *C, const int *assign, int N, int K){
    double *sum_global = (double*)calloc(K, sizeof(double));
    int *cnt_global = (int*)calloc(K, sizeof(int));

    #pragma omp parallel
    {
        double *sum_local = (double*)calloc(K, sizeof(double));
        int *cnt_local = (int*)calloc(K, sizeof(int));

        #pragma omp for // Seção não critica (cada thread faz a contagem para uma parte do array)
        for(int i=0;i<N;i++){
            int a = assign[i];
            cnt_local[a] += 1;
            sum_local[a] += X[i];
        }

        #pragma omp critical // Seção critica, que soma os valores locais no array de posições global
        {
            for(int c=0;c<K;c++){
                sum_global[c] += sum_local[c];
                cnt_global[c] += cnt_local[c];
            }
        }

        free(sum_local);
        free(cnt_local);
    }

    for(int c=0;c<K;c++){
        if(cnt_global[c] > 0) C[c] = sum_global[c] / (double)cnt_global[c];
        else                  C[c] = X[0];
    }

    free(sum_global); free(cnt_global);
}
/* A função atualiza o valor das centroides, com base nos pontos */

static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d(X, C, assign, N, K);
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}
/* Função orquestradora */

/* ---------- main ---------- */
int main(int argc, char **argv){
    omp_set_num_threads(8);
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));

    double t0 = omp_get_wtime();
    int iters = 0; double sse = 0.0;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    double t1 = omp_get_wtime();

    printf("K-means 1D (OpenMP)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.8f s\n", iters, sse, t1 - t0);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}
