import numpy as np
from preprocessamento import min_max_fit, min_max_transform

def distancia(A, q, metrica="euclidiana", p=3, norma_A=None):
  if metrica == "euclidiana":
    return np.sqrt(np.sum((A-q)**2, axis = 1))
  if metrica == "manhattan":
    return np.sum(np.abs(A - q), axis=1)
  if metrica == "chebyshev":
    return np.max(np.abs(A - q), axis=1)
  if metrica == "minkowski":
    return np.sum(np.abs(A - q)**p, axis=1)**(1.0/p)
  if metrica == "cosseno":
    if norma_A is None:
      norma_A = np.sqrt(np.sum(A**2, axis=1))
    norma_q = np.sqrt(np.sum(q**2))
    similaridade = (A @ q) / (norma_A * norma_q + 1e-10)
    return 1.0 - similaridade  

  raise ValueError(f"metrica desconhecida: {metrica}")

class KNN:
  def __init__(self, k=5, metrica="euclidiana", ponderado=False, p=3):
    self.k = k
    self.metrica = metrica
    self.ponderado = ponderado
    self.p = p
    self._norma_treino = None

  def fit(self, X, y):
    self.x_treino = np.asarray(X, dtype=float)
    self.y_treino = np.asarray(y)
    self.classes = np.unique(y)
    if self.metrica == "cosseno":
      self._norma_treino = np.sqrt(np.sum(self.x_treino**2, axis=1))
    else:
      self._norma_treino = None
    return self

  def classificar_um(self, q):
      d = distancia(self.x_treino, q, self.metrica, self.p, self._norma_treino)
      if self.k < len(d):
        viz = np.argpartition(d, self.k)[:self.k]
      else:
        viz = np.arange(len(d))

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

def matriz_confusao(y_real, y_pred, n_classes):
  M = np.zeros((n_classes, n_classes), dtype=int)
  for real, pred in zip(y_real, y_pred):
    M[real, pred] += 1
  return M

def relatorio_metricas(M):
  total = M.sum()
  acuracia = np.trace(M) / total

  precisao, revocacao, f1 = [], [], []
  for c in range(M.shape[0]):
    VP = M[c, c]
    FP = M[:, c].sum() - VP
    FN = M[c, :].sum() - VP

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

