# pré processamento, normalização
import numpy as np
import pandas as pd


# parte do one hot encoding para dá um jeito nas nominais
def codificar(x, mapa_de_tipos):
  partes = []
  for coluna in x.columns:
    if mapa_de_tipos[coluna] == 'nominal':
      # converte variaveis categoricas em variaveis dummy (0 ou 1)
      dummies = pd.get_dummies(x[coluna], prefix=coluna)
      partes.append(dummies)
    elif mapa_de_tipos[coluna] == 'ordinal':
      partes.append(x[[coluna]])
    elif mapa_de_tipos[coluna] == 'continuo':
      partes.append(x[[coluna]])

  # axis = 1 para junção horizontal pelas colunas
  return pd.concat(partes, axis=1).astype(float)

# (x - min) / (max - min)
def min_max_fit(x):
  x = np.array(x, dtype=float)
  xMin = np.min(x, axis=0)
  xMax = np.max(x, axis=0)
  faixa = xMax - xMin
  faixa[faixa == 0] = 1.0

  return xMin, faixa

def min_max_transform(x, xMin, faixa):
  x = np.array(x, dtype=float)
  resul = (x - xMin) / faixa
  return resul

# z-score: (x - media) / desvio
def z_score_fit(x):
  x = np.array(x, dtype=float)
  media = np.mean(x, axis=0)
  desvio = np.std(x, axis=0)
  desvio[desvio == 0] = 1.0

  return media, desvio

def z_score_transform(x, media, desvio):
  x = np.array(x, dtype=float)
  resul = (x - media) / desvio
  return resul

def normalizar(x_treino, x_teste, metodo):
  if metodo == 'minmax':
      params = min_max_fit(x_treino)
      x_treino_norm = min_max_transform(x_treino, *params)
      x_teste_norm  = min_max_transform(x_teste,  *params)
  elif metodo == 'zscore':
    params = z_score_fit(x_treino)
    x_treino_norm = z_score_transform(x_treino, *params)
    x_teste_norm  = z_score_transform(x_teste,  *params)
  else:
    raise ValueError("metodo deve ser 'minmax' ou 'zscore'")
  
  return x_treino_norm, x_teste_norm, params