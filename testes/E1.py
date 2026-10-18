# experimentos e resultados

# E1 - Efeito da normalização: tabela 2 × 2 (árvore / k-NN × com / sem normalização)
# e explicação do porquê do resultado;

import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao sys.path
DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from dados import carregar_dados, separar_treino_teste
from preprocessamento import construir_atributos
from knn import KNN
from knn_teste import validacao_cruzada

if __name__ == "__main__":
    SEMENTE = 42
    K_ESCOLHIDO = 19      # k ótimo encontrado na validação cruzada do k-NN (no knn_teste)
    PROF_ESCOLHIDA = 6   # profundidade ótima encontrada na validação cruzada da Árvore

    print("Carregando dados (Cenário A)...")
    caminho_csv = str(DIRETORIO_RAIZ / "data.csv")
    X, y = carregar_dados(caminho_csv, cenario="A")
    classes, y_numerico = np.unique(y, return_inverse=True)

    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(
        X,
        y_numerico,
        seed=SEMENTE
    )

    # Engenharia de atributos
    X_treino = construir_atributos(X_treino)
    X_teste = construir_atributos(X_teste)

    print("\nExecutando Experimento E1 (Validação Cruzada 10-fold x 3 repetições)...")

    # ==========================================
    # 1. Árvore de Decisão (com e sem normalização)
    # ==========================================
    print("-> Avaliando Árvore de Decisão...")
    arvore_sem = DecisionTreeClassifier(
        max_depth=PROF_ESCOLHIDA,
        min_samples_leaf=5,
        random_state=SEMENTE
    )
    scores_arvore_sem = validacao_cruzada(
        arvore_sem,
        X_treino,
        y_treino,
        n_folds=10,
        n_repeticoes=3,
        semente=SEMENTE,
        metodo_norm=None
    )

    arvore_com = DecisionTreeClassifier(
        max_depth=PROF_ESCOLHIDA,
        min_samples_leaf=5,
        random_state=SEMENTE
    )
    scores_arvore_com = validacao_cruzada(
        arvore_com,
        X_treino,
        y_treino,
        n_folds=10,
        n_repeticoes=3,
        semente=SEMENTE,
        metodo_norm="minmax"
    )

    # ==========================================
    # 2. k-NN (com e sem normalização)
    # ==========================================
    print("-> Avaliando k-NN...")
    knn_sem = KNN(k=K_ESCOLHIDO, metrica="euclidiana", ponderado=True)
    scores_knn_sem = validacao_cruzada(
        knn_sem,
        X_treino,
        y_treino,
        n_folds=10,
        n_repeticoes=3,
        semente=SEMENTE,
        metodo_norm=None
    )

    knn_com = KNN(k=K_ESCOLHIDO, metrica="euclidiana", ponderado=True)
    scores_knn_com = validacao_cruzada(
        knn_com,
        X_treino,
        y_treino,
        n_folds=10,
        n_repeticoes=3,
        semente=SEMENTE,
        metodo_norm="minmax"
    )

    # Médias e desvios
    med_arvore_sem, std_arvore_sem = np.mean(scores_arvore_sem), np.std(scores_arvore_sem)
    med_arvore_com, std_arvore_com = np.mean(scores_arvore_com), np.std(scores_arvore_com)

    med_knn_sem, std_knn_sem = np.mean(scores_knn_sem), np.std(scores_knn_sem)
    med_knn_com, std_knn_com = np.mean(scores_knn_com), np.std(scores_knn_com)

    # ==========================================
    # 3. Exibição da Tabela 2x2 do Experimento E1
    # ==========================================
    print("\n" + "=" * 76)
    print("TABELA 2x2: RESULTADO DO EXPERIMENTO E1 (Acurácia ± Desvio Padrão)")
    print("=" * 76)
    print(f"{'Modelo':<25} | {'Sem Normalização':<22} | {'Com Normalização (Min-Max)':<22}")
    print("-" * 76)
    print(f"{f'Árvore (prof={PROF_ESCOLHIDA})':<25} | {med_arvore_sem:.4f} ± {std_arvore_sem:.4f}        | {med_arvore_com:.4f} ± {std_arvore_com:.4f}")
    print(f"{f'k-NN (k={K_ESCOLHIDO})':<25} | {med_knn_sem:.4f} ± {std_knn_sem:.4f}        | {med_knn_com:.4f} ± {std_knn_com:.4f}")
    print("-" * 76)


    dif_knn = (med_knn_com - med_knn_sem) * 100
    dif_arvore = (med_arvore_com - med_arvore_sem) * 100

    print("\nANÁLISE E CONCLUSÃO EXPERIMENTAL:")
    print(f"1. k-NN: Variação de {dif_knn:+.2f} p.p. com a normalização.")
    print("   -> A normalização é DECISIVA para o k-NN, pois o cálculo de distâncias")
    print("      euclidianas é sensível à escala dos atributos. Sem normalização,")
    print("      atributos com maior amplitude numérica dominam os cálculos.")
    print(f"\n2. Árvore de Decisão: Variação de {dif_arvore:+.2f} p.p. com a normalização.")
    print("   -> A normalização é IRRELEVANTE para a Árvore de Decisão, pois as")
    print("      divisões dependem apenas da ordenação dos valores de cada atributo.")
    print("      Transformações monótonas como Min-Max preservam a ordem relativa.")
    print("=" * 76)
