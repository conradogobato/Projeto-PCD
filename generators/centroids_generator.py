import random
import csv

def amostra_inteiros(min_val=1, max_val=150, quantidade=50, seed=None):
    """
    Gera 'quantidade' números inteiros únicos no intervalo fechado [min_val, max_val].
    """
    if seed is not None:
        random.seed(seed)

    total_possiveis = max_val - min_val + 1
    if quantidade > total_possiveis:
        raise ValueError("quantidade maior que o total de valores possíveis no intervalo.")

    return random.sample(range(min_val, max_val + 1), quantidade)

if __name__ == "__main__":
    # Exemplo: 50 inteiros únicos entre 1 e 150
    numeros = amostra_inteiros(min_val=1, max_val=20000000, quantidade=90000, seed=42)

    # Salva em dados_inteiros.csv (uma coluna: valor)
    with open("centroids_random_grande.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["valor"])
        for x in sorted(numeros):
            writer.writerow([x])

    print("Arquivo 'dados_inteiros.csv' gerado com sucesso.")
