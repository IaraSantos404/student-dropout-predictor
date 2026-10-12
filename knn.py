import numpy as np
from preprocessamento import min_max_fit, min_max_transform

def distancia(A, q, metrica="euclidiana", p=3):
  if metrica == "euclidiana":
    return np.sqrt(np.sum((A-q)**2, axis = 1))
  if metrica == "manhattan":
    return np.sum(np.abs(A - q), axis=1)
  if metrica == "chebyshev":
    return np.max(np.abs(A - q), axis=1)
  if metrica == "minkowski":
    return np.sum(np.abs(A - q)**p, axis=1)**(1.0/p)
  # if metrica == "cosseno":
  #   numerador =""
  #   denominador =""
  #   return 1.0 - numerador / denominador

  raise ValueError(f"metrica desconhecida: {metrica}")

class KNN:
  # k = quanitdade de vizinhos, p para distancia de Minkowski, ponderado para ponderar votos
  def __init__(self, k=5, metrica="euclidiana", ponderado=False, p=3):
    self.k = k
    self.metrica = metrica
    self.ponderado = ponderado
    self.p = p

  # armazena conjunto de treino e seus rótulos, além das classes presentes no conjunto de treino
  def fit(self, X, y):
    self.x_treino = X
    self.y_treino = y
    self.classes = np.unique(y)
    return self
  
  # calcula a distancia entre ponto q e todos os exemplos do conjunto de treino
  # equivalente ao classificar_knn 
  def classificar_um(self, q):
      d = distancia(self.x_treino, q, self.metrica, self.p)
      #usado para obter os indices dos k vizinhos mais próximos
      viz = np.argsort(d)[:self.k]

      # calcula caso seja ponderado, o epsilon é para evitar divisão por zero
      if self.ponderado:
        pesos = 1.0 / (d[viz] + 1e-10)
      else:
        pesos = np.ones(self.k)

      votos = np.zeros(len(self.classes))
      for i, classe in enumerate(self.classes):
        mascara = self.y_treino[viz] == classe
        votos[i] = np.sum(pesos[mascara])

      return self.classes[np.argmax(votos)]

  def predict(self, X):
    X = np.array(X, dtype=float)
    return np.array([self.classificar_um(q) for q in X], dtype=int)


# onde calculos como matriz de confusão,
# n_classes = n° de classes (dropout, enrolled, graduate) 
def matriz_confusao(y_real, y_pred, n_classes):
  # matriz quadrada de zeros, onde linhas = classes reais e colunas = classes previstas
  # tranforma em matriz quadrada de zeros
  M = np.zeros((n_classes, n_classes), dtype=int)
  # zip é função de gerar tuplas, onde cada tupla é formada por um elemento de y_real e um elemento de y_pred
  for real, pred in zip(y_real, y_pred):
    # soma 1 na posição da matriz correspondente a classe real e a classe prevista
    M[real, pred] += 1
  return M

# precisão: dos exemplos que o modelo previu como positivos, quantos realmente eram positivos (TP/(TP+FP))
# revocação: dos exemplos positivos, quantos o modelo conseguiu prever como positivos (TP/(TP+FN))
# f1-score: média harmônica entre precisão e revocação (2*P*R/(P+R))
def relatorio_metricas(M):
  total = M.sum()
  # total de acertos / total de exemplos (diagonal principal/total de elementos)
  acuracia = np.trace(M) / total

  precisao, revocacao, f1 = [], [], []
  for c in range(M.shape[0]):
    VP = M[c, c] # verdadeiros positivos - diagonal principal
    FP = M[:, c].sum() - VP # falsos positivos - soma das linhas da coluna c menos os verdadeiros positivos (1+0+1 para c = 0)
    FN = M[c, :].sum() - VP # falsos negativos - soma da linha menos os verdadeiros positivos

    P = VP / (VP + FP) if (VP + FP) > 0.0 else 0.0
    R = VP / (VP + FN) if (VP + FN) > 0.0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0.0 else 0.0

    precisao.append(P)
    revocacao.append(R)
    f1.append(F)

  return {
  "acuracia": acuracia,
  "precisao": np.array(precisao),
  "revocacao": np.array(revocacao),
  "f1": np.array(f1),
  "f1_macro": np.mean(f1),
  "acuracia_balanceada": np.mean(revocacao),
  }
