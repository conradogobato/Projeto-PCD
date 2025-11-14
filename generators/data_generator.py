import random
import csv
import math

def amostra_floats_aleatorios(min_val=0.0, max_val=150.0, quantidade=50, n_decimals=None, seed=None):
    """
    Gera 'quantidade' números float aleatórios e únicos no intervalo [min_val, max_val].
    - n_decimals: se None, não arredonda (quase impossível repetir; ainda assim garantimos unicidade).
                  se um inteiro >= 0, arredonda para esse número de casas decimais.
    """
    if seed is not None:
        random.seed(seed)

    if n_decimals is not None:
        # Número máximo de valores distintos possíveis com arredondamento
        max_distintos = int(math.floor((max_val - min_val) * (10 ** n_decimals))) + 1
        if quantidade > max_distintos:
            raise ValueError(
                f"quantidade ({quantidade}) > máximo distinto possível ({max_distintos}) "
                f"para n_decimals={n_decimals} no intervalo."
            )

    valores = set()
    while len(valores) < quantidade:
        x = random.uniform(min_val, max_val)
        if n_decimals is not None:
            x = round(x, n_decimals)
        valores.add(x)  # set garante unicidade

    return list(valores)

if __name__ == "__main__":
    # Exemplo 1: 50 valores aleatórios únicos SEM arredondar (alta entropia)
    numeros = amostra_floats_aleatorios(min_val=0.0, max_val=150000000.0, quantidade=160000, n_decimals=None, seed=42)

    # (Opcional) Exemplo 2: 50 valores com 2 casas decimais
    # numeros = amostra_floats_aleatorios(min_val=0.0, max_val=150.0, quantidade=50, n_decimals=2, seed=42)

    # Salva em CSV (uma coluna: valor) — ordem aleatória (não classifica)
    with open("dados_random_grande.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for x in numeros:
            writer.writerow([x])

    print("Arquivo 'dados_random.csv' gerado com sucesso.")
