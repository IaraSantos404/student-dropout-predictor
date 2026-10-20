# Predição de Evasão Estudantil no Ensino Superior (PP01)

---

## 1. Apresentação e Objetivo do Projeto

O objetivo deste projeto é construir, do dado bruto à recomendação final para a coordenação acadêmica, um pipeline experimental completo de Aprendizado de Máquina voltado à identificação precoce de estudantes em risco de evasão universitária. O sistema classifica cada aluno entre três categorias:
* **Dropout** (evadido)
* **Enrolled** (ainda matriculado / sem desfecho definitivo)
* **Graduate** (concluinte / formado)

O foco central da pesquisa experimental não é simplesmente maximizar a acurácia global, mas responder a três perguntas fundamentais de negócio e modelagem:
1. **Com que antecedência é possível prever a evasão sem trapacear com informação do futuro?** (Análise dos Cenários temporais A, B e C para evitar *data leakage*).
2. **Quais características realmente carregam sinal preditivo?** (Engenharia de atributos, seleção e extração de características via PCA).
3. **Qual modelo entregar à coordenação acadêmica?** (Avaliação multicritério considerando desempenho, interpretabilidade, viés ético e custo computacional).

### Restrição de Modelos
Conforme estabelecido no regulamento do PP01 (Seção 2), são utilizados **exclusivamente**:
* **Árvore de Decisão (ID3 / CART)** — com núcleo implementado do zero.
* **k-NN (k-Nearest Neighbors)** — com núcleo implementado do zero.

*Nota: Modelos das Unidades 5 a 8 (Naive Bayes, Redes Neurais, SVM, Random Forest, XGBoost, etc.) pertencem ao PP02 e não foram utilizados.*

---

## 2. A Base de Dados e Obtenção

* **Fonte:** UCI Machine Learning Repository — Dataset 697: *Predict Students' Dropout and Academic Success*  
* **DOI:** [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)  
* **Origem:** Instituto Politécnico de Portalegre (Portugal), registros de 2008 a 2019.  
* **Dimensões:** 4.424 instâncias e 36 atributos preditivos originais, sem valores ausentes declarados.  
* **Distribuição do Alvo (`Target`):**
  * `Graduate`: 2.209 instâncias ($\approx 49,9\%$)
  * `Dropout`: 1.421 instâncias ($\approx 32,1\%$)
  * `Enrolled`: 794 instâncias ($\approx 17,9\%$)

### ⚠️ Como Obter a Base de Dados 


Para executar o pipeline:
1. Acesse o repositório oficial da UCI: [https://archive.ics.uci.edu/dataset/697](https://archive.ics.uci.edu/dataset/697).
2. Baixe o arquivo de dados original (`data.csv`, delimitado por ponto e vírgula `;`).
3. Coloque o arquivo `data.csv` diretamente na **pasta raiz** deste projeto.

---

## 3. Estrutura do Repositório

```text
├── main.py                     # Script principal: orquestra e executa o pipeline em comando único
├── dados.py                    # Carregamento da base, particionamento e definição dos cenários A, B e C
├── preprocessamento.py         # Mapeamento de tipos, codificação One-Hot, norm. Min-Max e Z-Score
├── pca.py                      # Extração por PCA manual a partir da matriz de covariância (eigh)
├── arvoreDecisao.py            # Árvore de Decisão do zero (Entropia, Gini, regras SE-ENTÃO e poda)
├── knn.py                      # k-NN do zero (5 métricas de distância, voto majoritário e ponderado)
├── knn_teste.py                # Validação matemática do k-NN contra scikit-learn
├── arvoreDecisao_teste.py      # Validação matemática da Árvore contra scikit-learn e teste de sobreajuste
├── pca_testes.py               # Validação do PCA contra scikit-learn
├── requirements.txt            # Dependências com versões exatas do ambiente Python
├── .gitignore                  # Arquivos ignorados pelo Git (incluindo data.csv e ambiente virtual)
├── testes/                     # Módulos dos Experimentos Obrigatórios da Seção 6
│   ├── __init__.py             # Pacote de testes
│   ├── E1.py                   # E1: Efeito da normalização (Tabela 2x2)
│   ├── E2.py                   # E2: Análise de componentes principais (PCA)
│   ├── E3.py                   # E3: Medidas de distância e busca do k ótimo
│   ├── E4.py                   # E4: Custo computacional (Ansioso vs. Preguiçoso)
│   ├── E5.py                   # E5: Avaliação do modelo final no teste e diagnóstico de Enrolled
│   └── E6.py                   # E6: Interpretabilidade (regras) e análise crítica de viés por subgrupo
├── saidas_experimentos/        # Tabelas (.csv), figuras (.png) e regras dos experimentos E1, E4, E5 e E6
├── saidas_pca/                 # Cargas, scree plot, projeção 2D e curvas de desempenho do PCA
├── saidas_knn/                 # Gráficos de varredura de k e tabelas comparativas de distâncias
└── saidas_arvore/              # Regras textuais, gráficos de profundidade e cálculos manuais
```

---

## 4. Requisitos e Instalação

* **Linguagem:** Python 3.12 (compatível com $\ge$ 3.10)
* **Ambiente Virtual Recomendado:** `.venv`

### Passo a Passo de Instalação:

1. **Clonar o repositório:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd trabalho1
   ```

2. **Criar e ativar o ambiente virtual:**
   * **No Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * **No Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Posicionar a base de dados:**
   Certifique-se de que o arquivo `data.csv` baixado da UCI está na raiz do projeto.

---

## 5. Sementes Aleatórias (Reprodutibilidade)

Conforme a recomendação da Seção 6 da especificação:
* **Semente Aleatória Fixa:** `SEED = 42`
* Aplicada rigorosamente em:
  * Partição inicial Treino (80%) / Teste (20%) via `train_test_split(stratify=y, random_state=42)`.
  * Validação cruzada estratificada (`RepeatedStratifiedKFold(random_state=42)` e `StratifiedKFold(random_state=42)`).
  * Desempates determinísticos no PCA e classificadores de referência.

---

## 6. Instruções de Execução (Comando Único — Tópico 7.1)

O pipeline completo executa sem erros a partir de um **único comando** (por padrão, processa sequencialmente os cenários **A**, **B** e **C**):

```bash
python main.py
```

### Opções e Parâmetros Disponíveis:

O script suporta argumentos de linha de comando para execuções customizadas:

* **Executar em um cenário temporal específico (`A`, `B` ou `C`):**
  ```bash
  python main.py --cenario todos  # Executa A, B e C sequencialmente (Padrão)
  python main.py --cenario A      # No ato da matrícula
  python main.py --cenario B      # Fim do 1º semestre
  python main.py --cenario C      # Fim do 1º ano
  ```

* **Executar apenas um experimento específico:**
  ```bash
  python main.py --experimento E1   # Efeito da normalização
  python main.py --experimento E2   # Análise de PCA
  python main.py --experimento E3   # Medidas de distância
  python main.py --experimento E4   # Custo computacional
  python main.py --experimento E5   # Modelo final no teste
  python main.py --experimento E6   # Interpretabilidade e viés
  ```

* **Executar a validação matemática das implementações do zero:**
  ```bash
  python main.py --validar
  ```

* **Exibir a documentação e ajuda dos comandos:**
  ```bash
  python main.py --help
  ```

---

## 7. Os Três Cenários de Predição Temporal (Seção 3.1)

Para evitar vazamento temporal (*data leakage*), os modelos são avaliados em três momentos distintos:

| Cenário | Momento da Predição | Atributos Incluídos | Qtd. Atributos | Utilidade Prática |
| :---: | :--- | :--- | :---: | :--- |
| **A** | No ato da matrícula ($t_0$) | Demográficos, socioeconômicos, ingresso e macroeconômicos | 24 | **Máxima antecedência:** permite ações preventivas antes do início das aulas. |
| **B** | Fim do 1º semestre | Cenário A + unidades creditadas, inscritas, aprovadas e notas do 1º sem. | 30 | **Intermediário:** captura as primeiras reprovações do estudante. |
| **C** | Fim do 1º ano | Cenário B + todas as notas e aprovações do 2º semestre | 36 | **Alta acurácia:** melhor predição, porém tardia para evitar abandonos já ocorridos. |

---

## 8. Algoritmos Implementados do Zero e Validação

O núcleo dos classificadores foi implementado de forma autoral, sem dependência de bibliotecas de ML na indução:

1. **Extração de Características por PCA ([pca.py](file:///c:/Users/iarasantos/OneDrive/Documentos/Faculdade/Conteudos/6%C2%B0%20semestre/ML/trabalho1/pca.py)):**
   * Centralização dos dados $\bar{x}$, matriz de covariância empírica e decomposição espectral via `np.linalg.eigh`.
   * Projeção linear $Z = (X - \bar{x}) W_d$ com ordenação decrescente de autovalores.
   * **Validação:** Diferença máxima na variância explicada contra o `sklearn.decomposition.PCA` na ordem de $1.39 \times 10^{-17}$ (OK).

2. **Árvore de Decisão ID3/CART ([arvoreDecisao.py](file:///c:/Users/iarasantos/OneDrive/Documentos/Faculdade/Conteudos/6%C2%B0%20semestre/ML/trabalho1/arvoreDecisao.py)):**
   * Funções de impureza: Entropia de Shannon ($H = -\sum p \log_2 p$) e Índice Gini ($G = 1 - \sum p^2$).
   * Ganho de informação para limiares contínuos, parada por profundidade máxima (`prof_max`) e mínimo de amostras por folha (`min_folha`).
   * Extração de regras inteligíveis SE-ENTÃO com métricas de cobertura e confiança; algoritmo de poda pós-indução (`podar_arvore`).
   * **Validação:** Concordância de predições com o `DecisionTreeClassifier` do scikit-learn de 100,0% no cenário de validação.

3. **k-NN Ponderado ([knn.py](file:///c:/Users/iarasantos/OneDrive/Documentos/Faculdade/Conteudos/6%C2%B0%20semestre/ML/trabalho1/knn.py)):**
   * 5 métricas de distância vetoriais: Euclidiana, Manhattan, Cosseno, Chebyshev e Minkowski ($p=3$).
   * Votação majoritária e votação ponderada pelo inverso da distância ($w_i = \frac{1}{d_i + 10^{-10}}$).
   * **Validação:** Concordância de predições de 100,0% contra o `KNeighborsClassifier` do scikit-learn.

---

## 9. Experimentos Obrigatórios (Seção 6)

O pipeline executa e consolida os seguintes experimentos:

* **E1 — Efeito da Normalização:** Gera tabela 2×2 (Árvore / k-NN $\times$ com / sem Min-Max). Demonstra empiricamente que a normalização é decisiva para o k-NN (+7.4 p.p.) e indiferente para a Árvore (0.00 p.p.).
* **E2 — Análise de Componentes Principais:** Gera gráfico de variância explicada acumulada (escolha de $d$ para 90% e 95%), cargas das 3 primeiras componentes, projeção 2D das classes e curva de acurácia vs. $d$ comparando Árvore e k-NN.
* **E3 — Medidas de Distância no k-NN:** Varredura sistemática de $k \in \{1, 3, 5, \dots, 51\}$ para as 5 métricas, identificando o $k$ ótimo e contrastando a robustez de Manhattan em dados mistos One-Hot.
* **E4 — Custo Computacional:** Mede o tempo de treino e predição de 1.000 instâncias. Contrasta o aprendizado ansioso (*Eager*) da Árvore com o aprendizado preguiçoso (*Lazy*) do k-NN e a aceleração proporcionada pela redução de dimensionalidade por PCA.
* **E5 — Modelo Final no Teste Independente:** Treina o modelo campeão no conjunto de treino (80%) e avalia uma única vez no teste (20%). Produz matriz de confusão heatmap, relatório de métricas por classe (P, R, F1) e diagnóstico da classe *Enrolled*.
* **E6 — Interpretabilidade e Análise Crítica de Viés:** Extrai regras de decisão textuais para a coordenação, calcula taxas de erro e disparidades em subgrupos sensíveis (Gênero, Condição de Bolsista e Situação Financeira/Devedor).

---

## 10. Protocolo Experimental e Métrica de Decisão (Seção 4.5)

* **Protocolo de Validação:** Validação Cruzada Estratificada 10-fold com 3 repetições (`RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)`).
* **Ausência de Vazamento (*Zero Leakage*):** Todos os passos de codificação One-Hot e normalização Min-Max/Z-Score são ajustados exclusivamente no conjunto de treino de cada fold.
* **Métrica Única de Decisão:** **F1-Macro**  
  * *Justificativa:* Diante do forte desbalanceamento das três classes da base (*Graduate* $\approx 50\%$, *Dropout* $\approx 32\%$ e *Enrolled* $\approx 18\%$), a acurácia tradicional favorece predições da classe majoritária. O F1-macro atribui peso igual ($1/K$) para o desempenho em cada uma das classes, penalizando modelos que ignoram as minoritárias.
* **Análise Secundária Binária:** Avaliação da tarefa binária (*Dropout* $\times$ Demais) com curva e cálculo de **AUC-ROC**.

---

## 11. Relação de Artefatos Gerados

Todas as saídas numéricas e visuais são salvas automaticamente em disco:

### Em `saidas_experimentos/`:
* `e1_efeito_normalizacao_<cenario>.csv`: Tabela 2x2 com médias e desvios-padrão de acurácia.
* `e4_custo_computacional_<cenario>.csv`: Tempos de treino e inferência (1.000 instâncias).
* `e5_matriz_confusao_<cenario>.png`: Heatmap da matriz de confusão no teste.
* `e5_metricas_teste_<cenario>.csv`: Precisão, revocação, F1-score e suporte por classe e média macro.
* `e6_taxa_erro_subgrupos_<cenario>.png`: Gráfico de barras da taxa de erro por subgrupo sensível.
* `e6_analise_vies_<cenario>.csv`: Disparidade de erro e acurácia por subgrupo.
* `e6_regras_decisao_<cenario>.txt`: Regras de decisão da árvore exportadas para a coordenação.

### Em `saidas_pca/`:
* `e2_cargas_com_padronizacao_<cenario>.csv` e `e2_cargas_sem_padronizacao_<cenario>.csv`: Cargas das 3 primeiras componentes.
* `e2_variancia_explicada_<cenario>.png`: Gráfico de variância individual e acumulada com corte de 90%.
* `e2_projecao_2d_<cenario>.png`: Dispersão 2D dos alunos (PC1 $\times$ PC2) colorida pelas classes.
* `e2_curva_desempenho_<cenario>.png` e `e2_desempenho_pca_<cenario>.csv`: Acurácia vs. dimensão $d$ para Árvore e k-NN.

### Em `saidas_knn/`:
* `e3_curva_k_metricas_<cenario>.png`: Curvas comparativas de desempenho em função de $k$ para as 5 distâncias.
* `e3_comparacao_distancias_<cenario>.csv`: Tabela consolidada com o $k$ ótimo e acurácia de cada métrica.

### Em `saidas_arvore/`:
* `profundidade_<cenario>.png` e `profundidade_<cenario>.csv`: Curva de treino vs. validação para detecção de sobreajuste.
* `calculo_manual_<cenario>.csv`: Validação de Entropia e Gini contra cálculo manual.
* `binario_<cenario>.csv`: Resultados da tarefa secundária binária com AUC-ROC.
* `regras_<cenario>.txt` e `regras_<cenario>.csv`: Regras com cobertura e confiança.

---

## 12. Equipe e Declaração de Uso de IA

* **Equipe de Desenvolvimento:** Alunos do Bacharelado em Ciência da Computação / Engenharia de Software (UFC Russas).
* **Declaração de Uso de Ferramentas de IA (Seção 2 e 7.2):**  
  Em conformidade com as regras de originalidade do PP01, ferramentas de Inteligência Artificial foram utilizadas como suporte e apoio à estruturação de scripts, orquestração de testes e otimização de código. Todas as implementações matemáticas dos algoritmos (PCA, Árvore de Decisão e k-NN), seleção de hiperparâmetros e análises críticas dos resultados foram revisadas, validadas e compreendidas integralmente pela equipe.
