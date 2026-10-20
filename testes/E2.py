# Experimento E2: Análise de Componentes Principais (PCA)
# - Efeito da padronização prévia (cargas das 3 primeiras componentes)
# - Gráfico de variância explicada acumulada e escolha de d
# - Projeção 2D colorida por classe (análise da sobreposição de Enrolled)
# - Curva de desempenho (Árvore e k-NN no mesmo gráfico) para d = 2, 5, 10, 20...

import sys
from pathlib import Path
import os

DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arvoreDecisao import ArvoreDecisao
from sklearn.model_selection import StratifiedKFold

from dados import carregar_dados, separar_treino_teste, MAPA_DE_TIPOS
from preprocessamento import preparar, construir_atributos
from pca import pca_ajustar, pca_projetar, cargas_componentes, pontuar_atributos, selecionar_topk
from knn import KNN

# Configurações gerais
plt.switch_backend('Agg')
SEMENTE = 42
CENARIO = 'C'
PASTA_SAIDA = os.path.join(str(DIRETORIO_RAIZ), 'saidas_pca')
os.makedirs(PASTA_SAIDA, exist_ok=True)

CORES_CLASSES = {'Dropout': '#D55E00', 'Enrolled': '#0072B2', 'Graduate': '#009E73'}

def executar_experimento_e2(cenario=CENARIO):
    print("=" * 76)
    print(f"EXPERIMENTO E2: ANÁLISE DE COMPONENTES PRINCIPAIS (CENÁRIO {cenario})")
    print("=" * 76)

    caminho_csv = str(DIRETORIO_RAIZ / "data.csv")
    X, y = carregar_dados(caminho_csv, cenario=cenario)
    classes, y_numerico = np.unique(y, return_inverse=True)

    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(
        X, y_numerico, seed=SEMENTE
    )

    X_treino = construir_atributos(X_treino)
    X_teste = construir_atributos(X_teste)

    mapa_tipos = {c: MAPA_DE_TIPOS.get(c, 'continuo') for c in X_treino.columns}

    # Prepara dados padronizados (Z-Score) e sem padronizar
    X_tr_std, X_te_std, nomes_atributos = preparar(X_treino, X_teste, mapa_tipos, metodo='zscore')
    X_tr_raw, X_te_raw, _ = preparar(X_treino, X_teste, mapa_tipos, metodo=None)

    # =========================================================================
    # PARTE 1: EFEITO DA PADRONIZAÇÃO PRÉVIA E CARGAS DAS 3 PRIMEIRAS COMPONENTES
    # =========================================================================
    print("\n[1] Efeito da Padronização Prévia:")
    # Sem padronização
    _, W_raw, ve_raw = pca_ajustar(X_tr_raw, 3)
    df_cargas_raw = cargas_componentes(W_raw, nomes_atributos, 3)

    # Com padronização (Z-score)
    media_std, W_std, ve_std = pca_ajustar(X_tr_std, None)
    df_cargas_std = cargas_componentes(W_std, nomes_atributos, 3)

    df_cargas_std.to_csv(os.path.join(PASTA_SAIDA, f'e2_cargas_com_padronizacao_{cenario}.csv'))
    df_cargas_raw.to_csv(os.path.join(PASTA_SAIDA, f'e2_cargas_sem_padronizacao_{cenario}.csv'))

    print("   -> Cargas salvas em CSV na pasta saidas_pca/.")
    print("   -> Principais variáveis na PC1 (COM padronização):")
    top_pc1 = df_cargas_std['PC1'].abs().sort_values(ascending=False).head(4)
    for col in top_pc1.index:
        print(f"      * {col}: peso = {df_cargas_std.loc[col, 'PC1']:+.4f}")

    # =========================================================================
    # PARTE 2: VARIÂNCIA EXPLICADA ACUMULADA E ESCOLHA DE d
    # =========================================================================
    print("\n[2] Gráfico de Variância Explicada (Critério 90% e 95%):")
    cum_ve = np.cumsum(ve_std)
    d_90 = int(np.argmax(cum_ve >= 0.90) + 1)
    d_95 = int(np.argmax(cum_ve >= 0.95) + 1)
    print(f"   -> PC1 explica {ve_std[0]:.2%} da variância.")
    print(f"   -> PC1 + PC2 explicam {cum_ve[1]:.2%} da variância.")
    print(f"   -> d para atingir 90% de variância: d = {d_90}")
    print(f"   -> d para atingir 95% de variância: d = {d_95}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_comp = np.arange(1, len(ve_std) + 1)
    ax.bar(x_comp[:40], ve_std[:40], alpha=0.5, color='#4A90E2', label='Variância Individual')
    ax.plot(x_comp[:40], cum_ve[:40], 'o-', color='#003366', lw=2, ms=3, label='Variância Acumulada')
    ax.axhline(0.90, ls='--', color='gray', label='Alvo 90%')
    ax.axvline(d_90, ls=':', color='#D55E00', lw=2, label=f'd={d_90} (90%)')
    ax.set_xlabel('Número de Componentes Principais (d)')
    ax.set_ylabel('Variância Explicada')
    ax.set_title(f'Variância Explicada pelo PCA - Cenário {cenario}')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho_fig_var = os.path.join(PASTA_SAIDA, f'e2_variancia_explicada_{cenario}.png')
    fig.savefig(caminho_fig_var, dpi=150)
    plt.close(fig)
    print(f"   -> Gráfico salvo em: {caminho_fig_var}")

    # =========================================================================
    # PARTE 3: PROJEÇÃO 2D COLORIDA POR CLASSE (PC1 x PC2)
    # =========================================================================
    print("\n[3] Projeção 2D Colorida por Classe (PC1 x PC2):")
    Z_tr_2d = pca_projetar(X_tr_std, media_std, W_std[:, :2])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for idx_c, nome_c in enumerate(classes):
        mascara = (y_treino == idx_c)
        ax.scatter(
            Z_tr_2d[mascara, 0],
            Z_tr_2d[mascara, 1],
            s=12,
            alpha=0.5,
            color=CORES_CLASSES.get(nome_c, 'gray'),
            label=nome_c
        )
    ax.set_xlabel(f'Componente Principal 1 ({ve_std[0]:.1%})')
    ax.set_ylabel(f'Componente Principal 2 ({ve_std[1]:.1%})')
    ax.set_title(f'Projeção 2D dos Alunos (PCA) - Cenário {cenario}')
    ax.legend(markerscale=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho_fig_2d = os.path.join(PASTA_SAIDA, f'e2_projecao_2d_{cenario}.png')
    fig.savefig(caminho_fig_2d, dpi=150)
    plt.close(fig)
    print(f"   -> Gráfico 2D salvo em: {caminho_fig_2d}")

    # =========================================================================
    # PARTE 4: CURVA DE DESEMPENHO (ÁRVORE E k-NN NO MESMO GRÁFICO)
    # =========================================================================
    print("\n[4] Avaliando Curva de Desempenho em Função de d (d = 2, 5, 10, 20...):")
    # Valores de d a testar (até o d_95 ou total disponível)
    max_d = min(X_tr_std.shape[1], 40)
    candidatos_d = [2, 5, 10, 15, 20, 25, 30]
    candidatos_d = [d for d in candidatos_d if d <= max_d]
    if d_90 not in candidatos_d and d_90 <= max_d:
        candidatos_d.append(d_90)
    candidatos_d.sort()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)

    res_arvore_media, res_arvore_std = [], []
    res_knn_pca_media, res_knn_pca_std = [], []
    res_knn_sel_media, res_knn_sel_std = [], []

    for d in candidatos_d:
        scores_arvore = []
        scores_knn_pca = []
        scores_knn_sel = []

        for tr_idx, val_idx in skf.split(X_treino, y_treino):
            X_tr_fold = X_treino.iloc[tr_idx]
            X_val_fold = X_treino.iloc[val_idx]
            y_tr_fold = y_treino[tr_idx]
            y_val_fold = y_treino[val_idx]

            # 1. Padroniza dentro do fold (sem vazamento)
            Xtr_s, Xval_s, _ = preparar(X_tr_fold, X_val_fold, mapa_tipos, metodo='zscore')

            # 2. Ajusta PCA no treino do fold e projeta (Extração)
            media_fold, W_fold, _ = pca_ajustar(Xtr_s, d)
            Z_tr_fold = pca_projetar(Xtr_s, media_fold, W_fold)
            Z_val_fold = pca_projetar(Xval_s, media_fold, W_fold)

            # 3. Seleção Top-k de atributos via ANOVA (Seleção)
            scores_feat = pontuar_atributos(Xtr_s, y_tr_fold, criterio='anova')
            top_k_idx = selecionar_topk(scores_feat, d)
            S_tr_fold = Xtr_s[:, top_k_idx]
            S_val_fold = Xval_s[:, top_k_idx]

            # 4. Árvore de Decisão (PCA)
            arvore = ArvoreDecisao(prof_max=6, min_folha=5)
            arvore.fit(Z_tr_fold, y_tr_fold)
            pred_arvore = arvore.predict(Z_val_fold)
            scores_arvore.append(np.mean(pred_arvore == y_val_fold))

            # 5. k-NN (PCA - Projeção)
            knn_pca = KNN(k=19, metrica="euclidiana", ponderado=True)
            knn_pca.fit(Z_tr_fold, y_tr_fold)
            pred_knn_p = knn_pca.predict(Z_val_fold)
            scores_knn_pca.append(np.mean(pred_knn_p == y_val_fold))

            # 6. k-NN (Top-k - Seleção)
            knn_sel = KNN(k=19, metrica="euclidiana", ponderado=True)
            knn_sel.fit(S_tr_fold, y_tr_fold)
            pred_knn_s = knn_sel.predict(S_val_fold)
            scores_knn_sel.append(np.mean(pred_knn_s == y_val_fold))

        med_a, std_a = np.mean(scores_arvore), np.std(scores_arvore)
        med_kp, std_kp = np.mean(scores_knn_pca), np.std(scores_knn_pca)
        med_ks, std_ks = np.mean(scores_knn_sel), np.std(scores_knn_sel)

        res_arvore_media.append(med_a)
        res_arvore_std.append(std_a)
        res_knn_pca_media.append(med_kp)
        res_knn_pca_std.append(std_kp)
        res_knn_sel_media.append(med_ks)
        res_knn_sel_std.append(std_ks)

        print(f"   * d = {d:2d} | Árvore (PCA): {med_a:.4f} ± {std_a:.4f} | k-NN (PCA): {med_kp:.4f} ± {std_kp:.4f} | k-NN (Seleção): {med_ks:.4f} ± {std_ks:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        candidatos_d, res_arvore_media, yerr=res_arvore_std,
        fmt='o-', color='#D55E00', capsize=4, lw=1.8, label='Árvore de Decisão (PCA, prof=6)'
    )
    ax.errorbar(
        candidatos_d, res_knn_pca_media, yerr=res_knn_pca_std,
        fmt='s--', color='#0072B2', capsize=4, lw=1.8, label='k-NN (Projeção PCA, k=19)'
    )
    ax.errorbar(
        candidatos_d, res_knn_sel_media, yerr=res_knn_sel_std,
        fmt='^-.', color='#009E73', capsize=4, lw=1.8, label='k-NN (Seleção Top-k ANOVA, k=19)'
    )
    ax.axvline(d_90, ls=':', color='gray', label=f'd={d_90} (90% Variância)')

    ax.set_xlabel('Número de Dimensões / Atributos (d)')
    ax.set_ylabel('Acurácia (Validação Cruzada 10-fold)')
    ax.set_title(f'Desempenho: Seleção vs. Projeção PCA (Cenário {cenario})')
    ax.set_xticks(candidatos_d)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    caminho_fig_curva = os.path.join(PASTA_SAIDA, f'e2_curva_desempenho_{cenario}.png')
    fig.savefig(caminho_fig_curva, dpi=150)
    plt.close(fig)
    print(f"\n   -> Gráfico da curva de desempenho salvo em: {caminho_fig_curva}")

    # Salva tabela de resultados em CSV
    df_res = pd.DataFrame({
        'd': candidatos_d,
        'acuracia_arvore_pca_media': np.round(res_arvore_media, 4),
        'acuracia_arvore_pca_std': np.round(res_arvore_std, 4),
        'acuracia_knn_pca_media': np.round(res_knn_pca_media, 4),
        'acuracia_knn_pca_std': np.round(res_knn_pca_std, 4),
        'acuracia_knn_selecao_media': np.round(res_knn_sel_media, 4),
        'acuracia_knn_selecao_std': np.round(res_knn_sel_std, 4),
    })
    caminho_csv_res = os.path.join(PASTA_SAIDA, f'e2_desempenho_pca_{cenario}.csv')
    df_res.to_csv(caminho_csv_res, index=False)
    print(f"   -> Tabela consolidada salva em: {caminho_csv_res}")

    print("\n" + "=" * 76)
    print("EXPERIMENTO E2 CONCLUÍDO COM SUCESSO!")

    print("=" * 76)

if __name__ == "__main__":
    executar_experimento_e2()
