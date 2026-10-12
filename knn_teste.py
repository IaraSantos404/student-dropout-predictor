import numpy as np

from dados import carregar_dados, separar_treino_teste
from preprocessamento import preparar, min_max_fit, min_max_transform
from dados import MAPA_DE_TIPOS
from knn import KNN, matriz_confusao, relatorio_metricas

X, y = carregar_dados("data.csv", cenario="A")

classes, y_numerico = np.unique(y, return_inverse=True)

print("Classes:")
for codigo, nome in enumerate(classes):
    print(f"{codigo}: {nome}")

# Separa treino e teste
X_treino, X_teste, y_treino, y_teste = separar_treino_teste(
    X,
    y_numerico,
    seed=42
)

# Mantém somente os tipos das colunas do cenário A
mapa_cenario_a = {
    coluna: MAPA_DE_TIPOS[coluna]
    for coluna in X_treino.columns
}

# Codifica variáveis nominais e normaliza com Min-Max
X_treino_processado, X_teste_processado, nomes_atributos = preparar(
    X_treino,
    X_teste,
    mapa_cenario_a,
    metodo="minmax"
)

print("\nFormato do treino:", X_treino_processado.shape)
print("Formato do teste:", X_teste_processado.shape)
print("Quantidade de atributos:", len(nomes_atributos))

# Treina o KNN simples
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

resultado = relatorio_metricas(M)
print("\nMatriz de confusão:")
print(M)
print("\nRelatório:")
for chave, valor in resultado.items():
    print(f"{chave}: {valor}")


# Validação usando o KNN do scikit-learn
from sklearn.neighbors import KNeighborsClassifier

K_ESCOLHIDO = 5
SEMENTE = 42

referencia = KNeighborsClassifier(
    n_neighbors=K_ESCOLHIDO,
    metric="euclidean",
    weights="distance"
)

# codigo de teste pra ver se o knn tava certo
# referencia.fit(X_treino_processado, y_treino)
# y_ref = referencia.predict(X_teste_processado)
# proporcao_iguais = np.mean(y_pred == y_ref)

# print("\nComparação com scikit-learn:")
# print("Acurácia própria:",
#       np.mean(y_pred == y_teste))

# print("Acurácia sklearn:",
#       np.mean(y_ref == y_teste))

# print("Proporção de previsões iguais:",
#       proporcao_iguais)

from sklearn.model_selection import train_test_split, StratifiedKFold

def cv_acuracia(X, y, k, metrica="euclidiana", n_folds=5):
  skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMENTE)
  scores = []
  for idx_tr, idx_val in skf.split(X, y):
    minimo_fold, faixa_fold = min_max_fit(X.iloc[idx_tr])
    Xtr = min_max_transform(X.iloc[idx_tr], minimo_fold, faixa_fold)
    Xva = min_max_transform(X.iloc[idx_val], minimo_fold, faixa_fold)

    modelo_fold = KNN(k=k, metrica=metrica)
    modelo_fold.fit(Xtr, y[idx_tr])
    yva_pred = modelo_fold.predict(Xva)
    scores.append(np.mean(yva_pred == y[idx_val]))

  return np.mean(scores), np.std(scores)

ks = list(range(1, 52, 2))
resultados_cv = [cv_acuracia(X_treino, y_treino, k) for k in ks]
medias_cv = [resultado[0] for resultado in resultados_cv]
desvios_cv = [resultado[1] for resultado in resultados_cv]
melhor_k = ks[int(np.argmax(medias_cv))]

print("Melhor k:", melhor_k)
for k, media, desvio in zip(ks, medias_cv, desvios_cv):
  print(k, media, desvio)