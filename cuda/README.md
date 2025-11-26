# K-Means 1D — CUDA

## Run on Google Colab
[text](https://colab.research.google.com/drive/1Ww9FaAR5-u5ku9YqC0yFFPVzcOSpNYZr)
### 1.
!nvcc -O2 -std=c++11 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda
### 2.
!./kmeans_1d_cuda dados_random.csv centroids_random.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

---

## Run on Ubuntu / Linux
### 1.
nvcc -O2 -std=c++11 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda
### 2.
./kmeans_1d_cuda dados_random.csv centroids_random.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]

---

## Run on Windows (WSL2 recomendado)
### 1.
nvcc -O2 -std=c++11 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda.exe
### 2.
./kmeans_1d_cuda.exe dados_random.csv centroids_random.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]


---

# Avaliação doe corretude

O algoritmo K-Means 1D em CUDA apresentou convergência estável em todos os experimentos realizados.  
Para todas as configurações testadas — incluindo diferentes números de blocos no grid — o valor final do erro quadrático total (SSE) foi idêntico:

**SSE final obtido: `1520953339877.944092`**

O SSE final, foi igual para todos os números de BlocosxThreads (com um número de 607 iterações onde o MAX_ITER era 2000). Isso indica que o resultado final do algoritmo é independente da configuração de paralelismo (desde que o número de threads seja suficiente para cobrir todos os pontos), sendo afetado apenas o tempo de execução.

### Convergência

O critério de parada baseado na variação relativa do SSE funcionou conforme esperado.  
Após algumas iterações, o valor do SSE estabilizou, e o algoritmo interrompeu antes de atingir o limite de iterações — evidenciando convergência numérica adequada para um eps de 0.000001.

### Impacto do Número de Blocos (Desempenho GPU)

Embora o SSE final permaneça o mesmo para qualquer configuração, o **tempo de execução variou conforme o número de blocos**.  
Quando o número de blocos excede a capacidade de execução simultânea da GPU (número de SMs × blocos por SM), ocorre *oversubscription*, fazendo a GPU executar blocos em múltiplas ondas — o que aumenta o tempo total.

Esse comportamento é documentado pela NVIDIA:

> *“Increasing the number of blocks beyond what can reside concurrently on the GPU does not increase parallel performance — excess blocks must wait for previous blocks to finish.”*  
> — **NVIDIA CUDA C Programming Guide, v12.0**, seção *Thread Hierarchy and Occupancy*.

Assim, mesmo que o resultado numérico seja sempre o mesmo, o tempo tende a aumentar quando o número de blocos ultrapassa o limite suportado simultaneamente pela GPU.

---
