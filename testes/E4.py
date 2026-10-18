# Experimento E4: Custo Computacional (Treino e Predição de 1.000 instâncias)
# - Contraste entre Aprendizado Ansioso (Árvore) e Preguiçoso (k-NN)
# - Efeito da Redução de Dimensionalidade (PCA) no tempo de predição do k-NN

import sys
from pathlib import Path
import os
import time

DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from dados import carregar_dados, separar_treino_teste, MAPA_DE_TIPOS
from preprocessamento import preparar, construir_atributos
from pca import pca_ajustar, pca_projetar
from knn import KNN

SEMENTE = 42
CENARIO = 'C'
PASTA_SAIDA = os.path.join(str(DIRETORIO_RAIZ), 'saidas_experimentos')
os.makedirs(PASTA_SAIDA, exist_ok=True)

def executar_experimento_e4():
    print("=" * 76)
    print(f"EXPERIMENTO E4: CUSTO COMPUTACIONAL (CENÁRIO {CENARIO})")
    print("=" * 76)

    # 1. Carregamento dos dados
    caminho_csv = str(DIRETORIO_RAIZ / "data.csv")
    X, y = carregar_dados(caminho_csv, cenario=CENARIO)
    classes, y_numerico = np.unique(y, return_inverse=True)

    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(
        X, y_numerico, seed=SEMENTE
    )

    X_treino = construir_atributos(X_treino)
    X_teste = construir_atributos(X_teste)

    mapa_tipos = {c: MAPA_DE_TIPOS.get(c, 'continuo') for c in X_treino.columns}

    # Pré-processamento com padronização
    X_tr_proc, X_te_proc, _ = preparar(X_treino, X_teste, mapa_tipos, metodo='minmax')

    # Garante exatamente 1.000 instâncias para teste replicando o conjunto de teste
    n_repeticoes = int(np.ceil(1000 / len(X_te_proc)))
    X_mil = np.tile(X_te_proc, (n_repeticoes, 1))[:1000]

    n_treino = len(X_tr_proc)
    d_original = X_tr_proc.shape[1]

    print(f"\nConjunto de treino: {n_treino} instâncias com {d_original} atributos.")
    print(f"Conjunto de predição: exatamente 1.000 instâncias.")

    # =========================================================================
    # 1. MODELO 1: ÁRVORE DE DECISÃO (Aprendizado Ansioso / Eager Learning)
    # =========================================================================
    print("\n[1] Medindo Árvore de Decisão...")
    modelo_arvore = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=SEMENTE)

    t0 = time.perf_counter()
    modelo_arvore.fit(X_tr_proc, y_treino)
    tempo_treino_arvore = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = modelo_arvore.predict(X_mil)
    tempo_pred_arvore = time.perf_counter() - t0

    # =========================================================================
    # 2. MODELO 2: k-NN com Todas as Variáveis (Aprendizado Preguiçoso / Lazy Learning)
    # =========================================================================
    print("[2] Medindo k-NN (D original = {} atributos)...".format(d_original))
    modelo_knn_original = KNN(k=19, metrica="euclidiana", ponderado=True)

    t0 = time.perf_counter()
    modelo_knn_original.fit(X_tr_proc, y_treino)
    tempo_treino_knn_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = modelo_knn_original.predict(X_mil)
    tempo_pred_knn_orig = time.perf_counter() - t0

    # =========================================================================
    # 3. MODELO 3: k-NN com Redução de Dimensionalidade por PCA (ex: d = 10)
    # =========================================================================
    d_reduzido = 10
    print(f"[3] Medindo k-NN com Redução de Dimensionalidade (PCA d = {d_reduzido})...")

    # Ajusta PCA no treino e projeta treino e as 1.000 instâncias
    media, W, _ = pca_ajustar(X_tr_proc, d_reduzido)
    Z_tr = pca_projetar(X_tr_proc, media, W)
    Z_mil = pca_projetar(X_mil, media, W)

    modelo_knn_pca = KNN(k=19, metrica="euclidiana", ponderado=True)

    t0 = time.perf_counter()
    modelo_knn_pca.fit(Z_tr, y_treino)
    tempo_treino_knn_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = modelo_knn_pca.predict(Z_mil)
    tempo_pred_knn_pca = time.perf_counter() - t0

    # =========================================================================
    # 4. TABELA DE COMPARAÇÃO DO EXPERIMENTO E4
    # =========================================================================
    tabela_resultados = [
        {
            'Modelo': 'Árvore de Decisão',
            'Paradigma': 'Ansioso (Eager)',
            'Dimensao': d_original,
            'Treino (ms)': tempo_treino_arvore * 1000,
            'Predição 1.000 inst. (ms)': tempo_pred_arvore * 1000
        },
        {
            'Modelo': 'k-NN (Original)',
            'Paradigma': 'Preguiçoso (Lazy)',
            'Dimensao': d_original,
            'Treino (ms)': tempo_treino_knn_orig * 1000,
            'Predição 1.000 inst. (ms)': tempo_pred_knn_orig * 1000
        },
        {
            'Modelo': f'k-NN (com PCA d={d_reduzido})',
            'Paradigma': 'Preguiçoso (Lazy)',
            'Dimensao': d_reduzido,
            'Treino (ms)': tempo_treino_knn_pca * 1000,
            'Predição 1.000 inst. (ms)': tempo_pred_knn_pca * 1000
        }
    ]

    df_res = pd.DataFrame(tabela_resultados)

    print("\n" + "=" * 76)
    print("TABELA E4: COMPARAÇÃO DE CUSTO COMPUTACIONAL (CENÁRIO C)")
    print("=" * 76)
    print(f"{'Modelo':<26} | {'Paradigma':<18} | {'Treino':<12} | {'Predição 1000 inst.':<18}")
    print("-" * 76)
    for linha in tabela_resultados:
        print(f"{linha['Modelo']:<26} | {linha['Paradigma']:<18} | {linha['Treino (ms)']:>8.2f} ms | {linha['Predição 1.000 inst. (ms)']:>15.2f} ms")
    print("-" * 76)

    # Salva em CSV
    caminho_csv_res = os.path.join(PASTA_SAIDA, f'e4_custo_computacional_{CENARIO}.csv')
    df_res.to_csv(caminho_csv_res, index=False)
    print(f"\n-> Tabela salva em: {caminho_csv_res}")

    # =========================================================================
    # 5. DISCUSSÃO E ANÁLISE TEÓRICA EXIGIDA NO ROTEIRO
    # =========================================================================
    aceleracao_knn = tempo_pred_knn_orig / tempo_pred_knn_pca
    fator_predicao = tempo_pred_knn_orig / tempo_pred_arvore

    print("\nDISCUSSÃO TEÓRICA:")
    print("1. Aprendizado Ansioso (Árvore de Decisão):")
    print(f"   * Gasta tempo no treino ({tempo_treino_arvore*1000:.2f} ms) induzindo a estrutura")
    print("     recursiva da árvore e encontrando limiares ótimos.")
    print(f"   * A predição é quase instantânea ({tempo_pred_arvore*1000:.2f} ms), pois requer apenas")
    print("     percorrer alguns nós (complexidade O(profundidade)).")
    print(f"\n2. Aprendizado Preguiçoso (k-NN):")
    print(f"   * O treino é instantâneo ({tempo_treino_knn_orig*1000:.2f} ms): apenas armazena os dados.")
    print(f"   * A predição é cara ({tempo_pred_knn_orig*1000:.2f} ms): precisa calcular distâncias contra")
    print(f"     todos os {n_treino} exemplos de treino para cada uma das 1.000 amostras.")
    print(f"   * Em inferência, a Árvore foi ~{fator_predicao:.1f}x mais rápida que o k-NN original!")
    print(f"\n3. Efeito da Redução de Dimensionalidade no k-NN:")
    print(f"   * Reduzindo de {d_original} para {d_reduzido} dimensões via PCA, o tempo de predição")
    print(f"     caiu de {tempo_pred_knn_orig*1000:.2f} ms para {tempo_pred_knn_pca*1000:.2f} ms (~{aceleracao_knn:.1f}x mais rápido).")
    print("   * Isso ocorre porque a complexidade de distância euclidiana é O(N * d). Reduzir d")
    print("     diminui diretamente o número de operações aritméticas por vizinho.")
    print("=" * 76)

if __name__ == "__main__":
    executar_experimento_e4()
