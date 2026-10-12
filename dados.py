import pandas as pd
from sklearn.model_selection import train_test_split

# classificação de tipos em nominal, ordinal e contínuo
MAPA_DE_TIPOS = {
  'Marital status': 'nominal',
  'Application mode': 'nominal',
  'Application order': 'ordinal',
  'Course': 'nominal',
  'Daytime/evening attendance': 'nominal',
  'Previous qualification': 'ordinal',
  'Previous qualification (grade)': 'continuo',
  'Nacionality': 'nominal',
  "Mother's qualification": 'ordinal',
  "Father's qualification": 'ordinal',
  "Mother's occupation": 'nominal',
  "Father's occupation": 'nominal',
  'Admission grade': 'continuo',
  'Displaced': 'nominal',
  'Educational special needs': 'nominal',
  'Debtor': 'nominal',
  'Tuition fees up to date': 'nominal',
  'Gender': 'nominal',
  'Scholarship holder': 'nominal',
  'Age at enrollment': 'continuo',
  'International': 'nominal',
  'Curricular units 1st sem (credited)' : 'continuo',
  'Curricular units 1st sem (enrolled)': 'continuo',
  'Curricular units 1st sem (evaluations)': 'continuo',
  'Curricular units 1st sem (approved)': 'continuo',
  'Curricular units 1st sem (grade)': 'continuo',
  'Curricular units 1st sem (without evaluations)': 'continuo',
  'Curricular units 2nd sem (credited)': 'continuo',
  'Curricular units 2nd sem (enrolled)': 'continuo',
  'Curricular units 2nd sem (evaluations)': 'continuo',
  'Curricular units 2nd sem (approved)': 'continuo',
  'Curricular units 2nd sem (grade)': 'continuo',
  'Curricular units 2nd sem (without evaluations)': 'continuo',
  'Unemployment rate': 'continuo',
  'Inflation rate': 'continuo',
  'GDP': 'continuo',
}

# Cenário A — no ato da matrícula: dados que existem antes de qualquer aula acontecer
# Cenário B — fim do 1º semestre: cenário A + o que o aluno rendeu no 1º sem
# Cenário C — fim do 1º ano: cenário B + o que o aluno rendeu no 2º sem
CENARIO_A = [
  'Marital status',
  'Application mode',
  'Application order',
  'Course',
  'Daytime/evening attendance',
  'Previous qualification',
  'Previous qualification (grade)',
  'Nacionality',
  "Mother's qualification",
  "Father's qualification",
  "Mother's occupation",
  "Father's occupation",
  'Admission grade',
  'Displaced',
  'Educational special needs',
  'Debtor',
  'Tuition fees up to date',
  'Gender',
  'Scholarship holder',
  'Age at enrollment',
  'International',
  'Unemployment rate',
  'Inflation rate',
  'GDP',
]


CENARIO_B = CENARIO_A + [
  'Curricular units 1st sem (credited)' ,
  'Curricular units 1st sem (enrolled)',
  'Curricular units 1st sem (evaluations)',
  'Curricular units 1st sem (approved)',
  'Curricular units 1st sem (grade)',
  'Curricular units 1st sem (without evaluations)',
]

CENARIO_C = CENARIO_B + [
  'Curricular units 2nd sem (credited)',
  'Curricular units 2nd sem (enrolled)',
  'Curricular units 2nd sem (evaluations)',
  'Curricular units 2nd sem (approved)',
  'Curricular units 2nd sem (grade)',
  'Curricular units 2nd sem (without evaluations)',
]

CENARIOS = {
  'A': CENARIO_A,
  'B': CENARIO_B,
  'C': CENARIO_C,
}

def carregar_dados(caminho='data.csv', cenario='A'):
  df = pd.read_csv(caminho, sep=';')
  df.columns = df.columns.str.strip()
  # x = todas as colunas de todos os cenários, y = coluna target - oq o modelo precisa prever
  x = df[CENARIOS[cenario]].copy()
  y = df['Target'].values
  return x, y

def separar_treino_teste(x, y, seed=42):
  return train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)

