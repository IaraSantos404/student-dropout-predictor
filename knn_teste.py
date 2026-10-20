import numpy as np

from dados import carregar_dados, separar_treino_teste
from preprocessamento import preparar, min_max_fit, min_max_transform, construir_atributos
from dados import MAPA_DE_TIPOS
from knn import KNN
from arvoreDecisao_teste import matriz_confusao, relatorio_metricas
from sklearn.model_selection import RepeatedStratifiedKFold

def validacao_cruzada(modelo, X, y, n_folds=10, n_repeticoes=3, semente=42, metodo_norm="minmax"):
  rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeticoes, random_state=semente)
  scores = []
  
  mapa_de_tipos = {coluna: MAPA_DE_TIPOS.get(coluna, 'continuo') for coluna in X.columns}

  for fold_idx, (idx_tr, idx_val) in enumerate(rskf.split(X, y)):
    X_tr_df, X_va_df = X.iloc[idx_tr], X.iloc[idx_val]
    y_tr_fold, y_va_fold = y[idx_tr], y[idx_val]

    Xtr, Xva, _ = preparar(X_tr_df, X_va_df, mapa_de_tipos, metodo=metodo_norm)

    modelo.fit(Xtr, y_tr_fold)
    y_pred = modelo.predict(Xva)
    
    acc = np.mean(y_pred == y_va_fold)
    scores.append(acc)
    
  return np.array(scores)

# Validação usando o KNN do scikit-learn
if __name__ == "__main__":
    X, y = carregar_dados("data.csv", cenario="A")

    classes, y_numerico = np.unique(y, return_inverse=True)

    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(
        X,
        y_numerico,
        seed=42
    )

    X_treino = construir_atributos(X_treino)
    X_teste = construir_atributos(X_teste)

    mapa_cenario_a = {
        coluna: MAPA_DE_TIPOS.get(coluna, 'continuo')
        for coluna in X_treino.columns
    }

    X_treino_processado, X_teste_processado, nomes_atributos = preparar(
        X_treino,
        X_teste,
        mapa_cenario_a,
        metodo="minmax"
    )

    modelo = KNN(
        k=5,
        metrica="euclidiana",
        ponderado=True
    )

    modelo.fit(X_treino_processado, y_treino)
    y_pred = modelo.predict(X_teste_processado)

    M = matriz_confusao(
        y_teste,
        y_pred,
        len(classes)
    )

    from sklearn.neighbors import KNeighborsClassifier
    resultado = relatorio_metricas(M)
    print("\nRelatório:")
    for chave, valor in resultado.items():
        print(f"{chave}: {valor}")

    K_ESCOLHIDO = 5
    SEMENTE = 42

    referencia = KNeighborsClassifier(
        n_neighbors=K_ESCOLHIDO,
        metric="euclidean",
        weights="distance"
    )

    #codigo de teste pra ver se o knn tava certo
    referencia.fit(X_treino_processado, y_treino)
    y_ref = referencia.predict(X_teste_processado)
    proporcao_iguais = np.mean(y_pred == y_ref)

    print("\nComparação com scikit-learn:")
    print("Acurácia própria:",
        np.mean(y_pred == y_teste))

    print("Acurácia sklearn:",
        np.mean(y_ref == y_teste))

    print("Proporção de previsões iguais:",
        proporcao_iguais)


    ks = list(range(1, 52, 2))
    medias_cv = []
    desvios_cv = []

    for k in ks:
        m = KNN(k=k, metrica="euclidiana", ponderado=True)
        scores_k = validacao_cruzada(m, X_treino, y_treino, n_folds=10, n_repeticoes=3, semente=SEMENTE)
        medias_cv.append(np.mean(scores_k))
        desvios_cv.append(np.std(scores_k))
        print(f"k = {k:2d}: acurácia = {np.mean(scores_k):.4f} desvio = {np.std(scores_k):.4f}")

    melhor_idx = int(np.argmax(medias_cv))
    print(f"\nMelhor k: {ks[melhor_idx]} ({medias_cv[melhor_idx]:.4f})")

