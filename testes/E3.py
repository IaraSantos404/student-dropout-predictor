# Experimento E3: Comparação de Medidas de Distância no k-NN
# - Varredura de k in {1, 3, 5, ..., 51} para cada métrica de distância
# - Identificação do k ótimo de cada uma
# - Comparação de desempenho (Acurácia +- Desvio) no k ótimo
# - Gráfico comparativo das curvas de k e discussão teórica para dados mistos

import sys
from pathlib import Path
import os

DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from dados import carregar_dados, separar_treino_teste, MAPA_DE_TIPOS
from preprocessamento import preparar, construir_atributos
from knn import KNN, distancia

# demora para achar k ótimo então seguem as métricas aqui:
# cenário C:
# -> Avaliando métrica: EUCLIDIANA...
#    Melhor k para euclidiana: k = 27 (Acurácia = 0.6985 ± 0.0180)

# -> Avaliando métrica: MANHATTAN...
#    Melhor k para manhattan: k = 27 (Acurácia = 0.7152 ± 0.0217)

# -> Avaliando métrica: COSSENO...
#    Melhor k para cosseno: k = 19 (Acurácia = 0.6731 ± 0.0185)

# Configurações
plt.switch_backend('Agg')
SEMENTE = 42
CENARIO = 'C'
PASTA_SAIDA = os.path.join(str(DIRETORIO_RAIZ), 'saidas_knn')
os.makedirs(PASTA_SAIDA, exist_ok=True)

METRICAS_TESTADAS = ['euclidiana', 'manhattan', 'cosseno', "chebyshev", "minkowski"]
CORES_METRICAS = {
    'euclidiana': '#0072B2',
    'manhattan': '#D55E00',
    'cosseno': '#009E73',
    'chebyshev': '#CC79A7',
    'minkowski': '#993399',
}

def executar_experimento_e3(cenario=CENARIO):
    print("=" * 76)
    print(f"EXPERIMENTO E3: MEDIDAS DE DISTÂNCIA NO k-NN (CENÁRIO {cenario})")
    print("=" * 76)

    # 1. Carregamento dos dados
    caminho_csv = str(DIRETORIO_RAIZ / "data.csv")
    X, y = carregar_dados(caminho_csv, cenario=cenario)
    classes, y_numerico = np.unique(y, return_inverse=True)

    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(
        X, y_numerico, seed=SEMENTE
    )

    X_treino = construir_atributos(X_treino)
    X_teste = construir_atributos(X_teste)

    mapa_tipos = {c: MAPA_DE_TIPOS.get(c, 'continuo') for c in X_treino.columns}

    # Valores de k ímpares para varredura
    valores_k = [1, 3, 5, 7, 9, 11, 15, 19, 23, 27, 31, 39, 45, 51]
    
    print("\n[1] Varrendo valores de k para encontrar o k ótimo de cada métrica:")
    print(f"    Valores de k testados: {valores_k}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)

    historico_metricas = {}
    melhores_k = {}
    max_k = max(valores_k)

    for metrica in METRICAS_TESTADAS:
        print(f"\n-> Avaliando métrica: {metrica.upper()}...")
        scores_k_map = {k: [] for k in valores_k}

        for tr_idx, val_idx in skf.split(X_treino, y_treino):
            X_tr_f = X_treino.iloc[tr_idx]
            X_val_f = X_treino.iloc[val_idx]
            y_tr_f = y_treino[tr_idx]
            y_val_f = y_treino[val_idx]

            # Normalização Min-Max sem vazamento
            Xtr_norm, Xval_norm, _ = preparar(X_tr_f, X_val_f, mapa_tipos, metodo='minmax')

            knn_base = KNN(k=max_k, metrica=metrica, ponderado=True)
            knn_base.fit(Xtr_norm, y_tr_f)

            preds_por_k = {k: [] for k in valores_k}
            for q in Xval_norm:
                d = distancia(knn_base.x_treino, q, knn_base.metrica, knn_base.p, knn_base._norma_treino)
                viz_max = np.argsort(d)[:max_k]
                pesos_max = 1.0 / (d[viz_max] + 1e-10)

                for k in valores_k:
                    viz_k = viz_max[:k]
                    pesos_k = pesos_max[:k]
                    votos = np.zeros(len(knn_base.classes))
                    for i_c, classe in enumerate(knn_base.classes):
                        votos[i_c] = np.sum(pesos_k[knn_base.y_treino[viz_k] == classe])
                    preds_por_k[k].append(knn_base.classes[np.argmax(votos)])

            for k in valores_k:
                acc_k = np.mean(np.array(preds_por_k[k]) == y_val_f)
                scores_k_map[k].append(acc_k)

        medias_k = [np.mean(scores_k_map[k]) for k in valores_k]
        desvios_k = [np.std(scores_k_map[k]) for k in valores_k]

        historico_metricas[metrica] = {
            'medias': medias_k,
            'desvios': desvios_k
        }

        melhor_idx = int(np.argmax(medias_k))
        melhores_k[metrica] = {
            'k_otimo': valores_k[melhor_idx],
            'acuracia': medias_k[melhor_idx],
            'desvio': desvios_k[melhor_idx]
        }
        print(f"   Melhor k para {metrica}: k = {valores_k[melhor_idx]} (Acurácia = {medias_k[melhor_idx]:.4f} ± {desvios_k[melhor_idx]:.4f})")

    # =========================================================================
    # 2. Gráfico Comparativo das Curvas de Desempenho em Função de k
    # =========================================================================
    print("\n[2] Gerando gráfico comparativo das curvas de k...")
    fig, ax = plt.subplots(figsize=(9, 5))

    for metrica in METRICAS_TESTADAS:
        cor = CORES_METRICAS.get(metrica, 'gray')
        ax.plot(
            valores_k,
            historico_metricas[metrica]['medias'],
            'o-',
            color=cor,
            lw=2,
            label=f"{metrica.capitalize()} (Melhor k={melhores_k[metrica]['k_otimo']})"
        )

    ax.set_xlabel('Valor de k (Número de Vizinhos)')
    ax.set_ylabel('Acurácia Média (10-fold CV)')
    ax.set_title(f'Desempenho do k-NN por Medida de Distância vs. k (Cenário {cenario})')
    ax.set_xticks(valores_k)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    caminho_fig = os.path.join(PASTA_SAIDA, f'e3_curva_k_metricas_{cenario}.png')
    fig.savefig(caminho_fig, dpi=150)
    plt.close(fig)
    print(f"   -> Gráfico salvo em: {caminho_fig}")

    # =========================================================================
    # 3. Tabela Comparativa Consolidada no k Ótimo
    # =========================================================================
    print("\n" + "=" * 76)
    print(f"TABELA E3: COMPARAÇÃO DAS MEDIDAS DE DISTÂNCIA NO k ÓTIMO (CENÁRIO {cenario})")
    print("=" * 76)
    print(f"{'Medida de Distância':<24} | {'k Ótimo':<10} | {'Acurácia (Média ± Desvio)':<28}")
    print("-" * 76)

    linhas_csv = []
    for metrica in METRICAS_TESTADAS:
        k_ot = melhores_k[metrica]['k_otimo']
        acc = melhores_k[metrica]['acuracia']
        std = melhores_k[metrica]['desvio']
        print(f"{metrica.capitalize():<24} | {k_ot:<10} | {acc:.4f} ± {std:.4f}")
        linhas_csv.append({
            'metrica': metrica,
            'k_otimo': k_ot,
            'acuracia_media': round(acc, 4),
            'acuracia_desvio': round(std, 4)
        })

    print("-" * 76)

    # Salva tabela em CSV
    caminho_csv_res = os.path.join(PASTA_SAIDA, f'e3_comparacao_distancias_{cenario}.csv')
    pd.DataFrame(linhas_csv).to_csv(caminho_csv_res, index=False)
    print(f"\n-> Tabela salva em: {caminho_csv_res}")



if __name__ == "__main__":
    executar_experimento_e3()
