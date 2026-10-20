# pré processamento, normalização
import numpy as np
import pandas as pd

# esse arquivo tem q implementar as funções de normalização
# one hot para variáveis nominais, etc
# para impedir vazamento de infromação, o ajuste deve ser feito apenas com os dados de treino
# por isso a separação entre funções de fit (só pra treino) e transform (treino e teste)

def codificar_fit(x, mapa_de_tipos, freq_min = 0.01):
  params = {}
  for coluna in x.columns:
    tipo = mapa_de_tipos.get(coluna, 'continuo')
    if tipo != 'nominal':
      params[coluna] = {'tipo': tipo}
      continue

    valores = sorted(x[coluna].unique())
    if len(valores) <= 2:
      params[coluna] = {'tipo': 'binaria', 'positivo': valores[-1]}
      continue

    freq = x[coluna].value_counts(normalize=True)
    mantidas = sorted(freq[freq >= freq_min].index.tolist())
    params[coluna] = {'tipo': 'nominal', 'categorias': mantidas, 'tem_outros': bool((freq < freq_min).any())}

  return params

def codificar_transform(x, params):
  partes = []
  for coluna, info in params.items():
    serie = x[coluna]

    if info['tipo'] in ['ordinal', 'continuo']:
      partes.append(serie.astype(float).to_frame(coluna))

    elif info['tipo'] == 'binaria':
      partes.append((serie == info['positivo']).astype(float).to_frame(coluna))

    else:
      cols = {}
      for cat in info['categorias']:
        cols[f'{coluna}_{cat}'] = (serie == cat).astype(float)
      if info['tem_outros']:
        cols[f'{coluna}_outros'] = (serie.isin(info['categorias']) == False).astype(float)
      partes.append(pd.DataFrame(cols, index=x.index))

  return pd.concat(partes, axis=1)

def construir_atributos(df):
  df = df.copy()

  # cenario A
  if 'Admission grade' in df.columns and "Previous qualification (grade)" in df.columns:
    df['delta_nota_ingresso'] = df['Admission grade'] - df['Previous qualification (grade)']
  
  if 'Debtor' in df.columns and 'Tuition fees up to date' in df.columns:
    df['risco_financeiro'] = ((df['Debtor'] == 1) & (df['Tuition fees up to date'] == 0)).astype(float)
  
  if "Mother's qualification" in df.columns and "Father's qualification" in df.columns:
    df['max_escolaridade_pais'] = np.maximum(df["Mother's qualification"], df["Father's qualification"]).astype(float)


  # cenario B 1°
  if 'Curricular units 1st sem (enrolled)' in df.columns:
    inscritas_1 = df['Curricular units 1st sem (enrolled)']
    aprovadas_1 = df['Curricular units 1st sem (approved)']
    sem_aval_1 = df['Curricular units 1st sem (without evaluations)']

    df['taxa_aprovacao_1sem'] = np.where(inscritas_1 > 0, aprovadas_1 / inscritas_1, 0.0)
    df['taxa_sem_avaliacao_1sem'] = np.where(inscritas_1 > 0, sem_aval_1 / inscritas_1, 0.0)

  # cenario C 2°
  if 'Curricular units 2nd sem (enrolled)' in df.columns:
    inscritas_2 = df['Curricular units 2nd sem (enrolled)']
    aprovadas_2 = df['Curricular units 2nd sem (approved)']
    sem_aval2 = df['Curricular units 2nd sem (without evaluations)']

    df['taxa_aprovacao_2sem'] = np.where(inscritas_2 > 0, aprovadas_2 / inscritas_2, 0.0)
    df['taxa_sem_avaliacao_2sem'] = np.where(inscritas_2 > 0, sem_aval2 / inscritas_2, 0.0)

    if "taxa_aprovacao_2sem" in df.columns:
      df['evolucao_aprovacao'] = df['taxa_aprovacao_2sem'] - df['taxa_aprovacao_1sem']

    if 'Curricular units 1st sem (approved)' in df.columns and 'Curricular units 2nd sem (approved)' in df.columns:
      df['evolucao_nota'] = (df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)'])
  
  return df


# NORMALIZACAO  
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

def preparar(x_treino, x_val, mapa_de_tipos, metodo=None, freq_min=0.01):
  params_cod = codificar_fit(x_treino, mapa_de_tipos, freq_min)
  xt = codificar_transform(x_treino, params_cod)
  xv = codificar_transform(x_val, params_cod)
  nomes = list(xt.columns)

  if metodo is None:
    return xt.values.astype(float), xv.values.astype(float), nomes

  xt_n, xv_n, _ = normalizar(xt.values, xv.values, metodo)
  return xt_n, xv_n, nomes



