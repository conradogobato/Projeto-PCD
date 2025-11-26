#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// ---- Importa arquivo e conta número de linhas como no serial
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); MPI_Abort(MPI_COMM_WORLD,1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int ws = 1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ ws=0; break; }
        }
        if(!ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    double *A = (double*)malloc(R*sizeof(double));
    FILE *f = fopen(path, "r");
    char line[8192];
    int r = 0;
    while(fgets(line,sizeof(line),f)){
        char *tok = strtok(line,",; \t");
        if(tok){ A[r++] = atof(tok); }
    }
    fclose(f);
    *n_out = R;
    return A;
}
// ----

// ---- Função de atribuição local
static double assignment_local(
    const double *Xloc, int nloc,
    const double *C, int K,
    int *assign_loc)
{
    double sse = 0.0;
    for(int i=0;i<nloc;i++){
        double xi = Xloc[i];
        double bestd = 1e300;
        int best = -1;
        for(int c=0;c<K;c++){
            double diff = xi - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign_loc[i] = best;
        sse += bestd;
    }
    return sse;
}
// ----

// Função de update global, que usa o MPIAllreduce
static void update_global(
    const double *Xloc, const int *assign_loc, int nloc,
    double *C, int K,
    double *tempo_allreduce)   // <<< ADICIONADO
{
    double *sum_loc = calloc(K, sizeof(double));
    int    *cnt_loc = calloc(K, sizeof(int));

    for(int i=0;i<nloc;i++){
        int a = assign_loc[i];
        sum_loc[a] += Xloc[i];
        cnt_loc[a] += 1;
    }

    double *sum_global = calloc(K, sizeof(double));
    int    *cnt_global = calloc(K, sizeof(int));

    // ---- MEDIÇÃO DO TEMPO DOS MPI_Allreduce ----
    double tA = MPI_Wtime();
    MPI_Allreduce(sum_loc, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(cnt_loc, cnt_global, K, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
    double tB = MPI_Wtime();
    *tempo_allreduce += (tB - tA);
    // --------------------------------------------

    for(int c=0;c<K;c++){
        if(cnt_global[c] > 0)
            C[c] = sum_global[c] / (double)cnt_global[c];
        else
            C[c] = C[c];  // mantém o centróide (mesma estratégia do naive)
    }

    free(sum_loc); free(cnt_loc);
    free(sum_global); free(cnt_global);
}
// ----

// ---- Função principal do K-means com MPI
double kmeans_1d_mpi(
    const double *Xloc, int nloc,
    double *C, int K,
    int max_iter, double eps,
    int rank,
    int *iter_out,
    double *tempo_allreduce)   // <<< ADICIONADO
{
    double prev_sse = 1e300;

    int *assign_loc = malloc(nloc*sizeof(int));
    int it;

    for(it=0; it<max_iter; it++){
        double sse_local = assignment_local(Xloc, nloc, C, K, assign_loc);

        double sse_global;
        MPI_Reduce(&sse_local, &sse_global,
                   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Bcast(&sse_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double rel = fabs(sse_global - prev_sse) /
                     (prev_sse > 0.0 ? prev_sse : 1.0);

        if(rel < eps){
            if(rank == 0)
                printf("Convergiu na iteração %d (SSE=%.6f)\n", it, sse_global);

            prev_sse = sse_global;
            break;
        }

        update_global(Xloc, assign_loc, nloc, C, K, tempo_allreduce);

        MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        prev_sse = sse_global;
    }

    free(assign_loc);
    *iter_out = it;
    return prev_sse;
}
// ----

// ---- Função main com MPI (TODAS AS LÓGICAS DE if rank == 0 SÃO PARA VERIFICAR SE É O 
//PROCESSO MASTER, QUE LE OS DADOS E IMPRIME OS RESULTADOS)
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if(argc < 3){
        if(rank==0)
            printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter] [eps]\n",
                    argv[0]);
        MPI_Finalize();
        return 0;
    }

    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;

    int N, K;
    double *X = NULL;
    double *C = NULL;

    if(rank == 0){
        X = read_csv_1col(argv[1], &N);
        C = read_csv_1col(argv[2], &K);
    }

    MPI_Bcast(&N, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT,    0, MPI_COMM_WORLD);

    if(rank != 0){
        C = malloc(K*sizeof(double));
    }
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int base = N / P;
    int rest = N % P;
    int nloc = base + (rank < rest ? 1 : 0);

    int offset = (rank < rest ? rank*(base+1) : rest + rank*base);

    double *Xloc = malloc(nloc*sizeof(double));

    if(rank == 0){
        for(int p=1; p<P; p++){
            int np = base + (p < rest ? 1 : 0);
            int offp = (p < rest ? p*(base+1) : rest + p*base);
            MPI_Send(X + offp, np, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        memcpy(Xloc, X + offset, nloc*sizeof(double));
    } else {
        MPI_Recv(Xloc, nloc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if(rank==0) printf("Rodando MPI K-means com %d processos...\n", P);

    int iter_final = 0;
    double tempo_allreduce = 0.0;  // <<< ADICIONADO

    double t0 = MPI_Wtime();
    double sse_final = kmeans_1d_mpi(Xloc, nloc, C, K, max_iter, eps, rank, &iter_final, &tempo_allreduce);
    double t1 = MPI_Wtime();

    if(rank==0){
        double tempo = (t1 - t0);
        printf("Tempo total: %.6f s\n", tempo);
        printf("SSE final: %.6f\n", sse_final);
        printf("Parou na iteração: %d\n", iter_final);
        printf("Tempo total gasto com MPI_Allreduce: %.6f s\n", tempo_allreduce);

        // ---- Salva os centróides finais em arquivo CSV ----
        FILE *fc = fopen("centroids_mpi.csv", "w");
        if(!fc){
            fprintf(stderr, "Erro ao abrir centroids_mpi.csv para escrita\n");
        } else {
            for(int c=0;c<K;c++)
                fprintf(fc, "%.6f\n", C[c]);
            fclose(fc);
        }

        // ---- Printar apenas resumo ----
        printf("Centróides salvos em centroids_mpi.csv\n");
    }

    free(C);
    free(Xloc);
    free(X);

    MPI_Finalize();
    return 0;
}
// ----
