import numpy as np 
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
import pandas as pd
# parte de caractesristicas

# seleção de caracteristicas
def pontuar_atributos( X, y, criterio):

    X = np.asarray(X, dtype=float)
 
    if criterio == 'mi':
        return mutual_info_classif(X, y, random_state=42)
 
    if criterio == 'anova':
        scores = f_classif(X, y)[0]
        # coluna constante gera F = NaN; nota 0 faz mais sentido
        return np.nan_to_num(scores, nan=0.0)
 
    if criterio == 'chi2':
        if (X < 0).any():
            raise ValueError("chi2 exige valores não negativos: "
                             "use os dados sem padronizar (ou colunas binárias/one-hot).")
        return np.nan_to_num(chi2(X, y)[0], nan=0.0)
 
    raise ValueError("criterio deve ser 'mi', 'anova' ou 'chi2'")

def selecionar_topk (scores , k):
    scores = np.asarray(scores)
    return np.argsort(scores)[::-1][:k]


# pca implementado na mão

def pca_ajustar(X_treino, n_componentes):
    X = np.asarray(X_treino, dtype=float)
 
    # 1) centralização
    media = X.mean(axis=0)
    Xc = X - media
 
    # 2) matriz de covariância (n_atributos x n_atributos)
    cov = np.atleast_2d(np.cov(Xc, rowvar=False))
 
    # 3) autovalores e autovetores (eigh: matriz simétrica, devolve em ordem crescente)
    autovalores, autovetores = np.linalg.eigh(cov)
 
    # 4) ordem decrescente por autovalor
    ordem = np.argsort(autovalores)[::-1]
    autovalores = np.clip(autovalores[ordem], 0.0, None)  # remove negativos minúsculos
    autovetores = autovetores[:, ordem]
 
    # sinal determinístico: o maior |valor| de cada autovetor fica positivo
    idx_max = np.argmax(np.abs(autovetores), axis=0)
    sinais = np.sign(autovetores[idx_max, np.arange(autovetores.shape[1])])
    sinais[sinais == 0] = 1.0
    autovetores = autovetores * sinais
 
    # 5) variância explicada de todas as componentes
    variancia_explicada = autovalores / autovalores.sum()
 
    # 6) fica só com as n_componentes primeiras
    if n_componentes is None:
        n_componentes = autovetores.shape[1]
    W = autovetores[:, :n_componentes]
 
    return media, W, variancia_explicada


def pca_projetar(X, media, W):
    X = np.asarray(X, dtype=float)
    return (X - media) @ W

def cargas_componentes(W, nome_atributos , n=3):
    n = min(n, W.shape[1])
    colunas = [f'PC{i + 1}' for i in range(n)]
    return pd.DataFrame(W[:, :n], index=list(nome_atributos), columns=colunas)








