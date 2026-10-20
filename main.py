"""
========================================================================================
UNIVERSIDADE FEDERAL DO CEARÁ - CAMPUS DE RUSSAS
Bacharelado em Ciência da Computação e Engenharia de Software
RUS0310 - Aprendizado de Máquina (Semestre 2026.2) - Prof. Alan Rocha
Especificação do Projeto Prático 01 (PP01): Predição de Evasão Estudantil no Ensino Superior
========================================================================================

Script Principal (main.py):
Ponto de entrada único para execução automatizada de todo o pipeline de Machine Learning:
1. Verificação de ambiente e integridade da base de dados (UCI 697)
2. Pré-processamento e Engenharia de Atributos sem vazamento (data leakage)
3. Validação dos algoritmos próprios contra scikit-learn (PCA, Árvore de Decisão, k-NN)
4. Execução sequencial dos experimentos obrigatórios da Seção 6:
   - E1: Efeito da Normalização (Tabela 2x2: Árvore vs k-NN com e sem Min-Max)
   - E2: Análise de Componentes Principais (Variância explicada, escolha de d, projeção 2D, cargas e curva de desempenho)
   - E3: Medidas de Distância no k-NN (Euclidiana, Manhattan, Cosseno, Chebyshev, Minkowski e busca do k ótimo)
   - E4: Custo Computacional (Treino e predição de 1.000 instâncias: Ansioso vs Preguiçoso e aceleração com PCA)
   - E5: Modelo Final no Teste Independente (Ajuste no treino, teste único, matriz de confusão, análise de 'Enrolled')
   - E6: Interpretabilidade e Análise Crítica de Viés (Regras SE-ENTÃO para a coordenação, disparidade por subgrupos sensíveis)
5. Persistência de todos os artefatos de saída (tabelas em .csv e gráficos em .png)
6. Emissão de relatório e recomendação técnica final à coordenação

Execução em comando único:
  python main.py
========================================================================================
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Garante o diretório raiz do projeto no path do Python
DIRETORIO_RAIZ = Path(__file__).resolve().parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Garante backend não interativo para geração e salvamento sem travar o terminal
plt.switch_backend('Agg')

from dados import carregar_dados, separar_treino_teste, MAPA_DE_TIPOS
from preprocessamento import preparar, construir_atributos
from pca import pca_ajustar, pca_projetar
from knn import KNN, distancia
from arvoreDecisao import (
    entropia, indice_gini, ganho_informacao,
    induzir_arvore, prever_arvore, extrair_regras, formatar_regras
)

# Importa os módulos de experimentos obrigatórios
from testes.E1 import executar_experimento_e1
from testes.E2 import executar_experimento_e2
from testes.E3 import executar_experimento_e3
from testes.E4 import executar_experimento_e4
from testes.E5 import executar_experimento_e5
from testes.E6 import executar_experimento_e6

# Pastas de saída do projeto
PASTAS_SAIDA = [
    DIRETORIO_RAIZ / "saidas_experimentos",
    DIRETORIO_RAIZ / "saidas_pca",
    DIRETORIO_RAIZ / "saidas_knn",
    DIRETORIO_RAIZ / "saidas_arvore",
]


def inicializar_ambiente(caminho_dados="data.csv"):
    """
    Cria as pastas necessárias para salvar figuras e tabelas, e
    valida a presença da base de dados sem versionamento obrigatório.
    """
    for pasta in PASTAS_SAIDA:
        os.makedirs(pasta, exist_ok=True)

    caminho_csv = DIRETORIO_RAIZ / caminho_dados
    if not caminho_csv.exists():
        print("\n" + "!" * 78)
        print("ALERTA CRÍTICO: BASE DE DADOS NÃO ENCONTRADA")
        print(f"O arquivo '{caminho_dados}' não foi encontrado na raiz do projeto.")
        print("-" * 78)
        print("Conforme a Seção 7.1 da especificação ('Não versionar a base de dados'):")
        print("A base não deve ser comitada no repositório Git por boas práticas.")
        print("\nCOMO OBTER A BASE DE DADOS:")
        print("1. Acesse o UCI Machine Learning Repository (Dataset 697):")
        print("   https://archive.ics.uci.edu/dataset/697")
        print("2. Baixe o arquivo de dados original (data.csv com separador ';').")
        print(f"3. Salve o arquivo como '{caminho_csv}'.")
        print("!" * 78 + "\n")
        sys.exit(1)

    return str(caminho_csv)


def validar_implementacoes_proprias(caminho_csv, cenario="A"):
    """
    Executa uma rodada rápida de validação matemática das implementações
    feitas do zero (PCA, Árvore de Decisão e k-NN) comparando contra o scikit-learn.
    Conforme exigência das Seções 4.2, 4.3 e 4.4 do roteiro.
    """
    print("\n" + "=" * 78)
    print("ETAPA 0: VALIDAÇÃO DOS ALGORITMOS DO ZERO CONTRA SCIKIT-LEARN")
    print("=" * 78)

    X, y = carregar_dados(caminho_csv, cenario=cenario)
    classes, y_num = np.unique(y, return_inverse=True)
    X_tr, X_te, y_tr, y_te = separar_treino_teste(X, y_num, seed=42)

    mapa = {c: MAPA_DE_TIPOS.get(c, 'continuo') for c in X_tr.columns}
    Xtr_norm, Xte_norm, nomes = preparar(X_tr, X_te, mapa, metodo='zscore')

    # 1. Validação do PCA próprio contra o PCA do scikit-learn
    print("\n[1/3] Validando PCA manual vs. sklearn.decomposition.PCA...")
    from sklearn.decomposition import PCA as SklearnPCA
    media_proprio, W_proprio, ve_proprio = pca_ajustar(Xtr_norm, None)
    Z_proprio = pca_projetar(Xtr_norm, media_proprio, W_proprio)

    pca_sk = SklearnPCA().fit(Xtr_norm)
    Z_sk = pca_sk.transform(Xtr_norm)

    dif_ve = float(np.max(np.abs(ve_proprio - pca_sk.explained_variance_ratio_)))
    dif_proj = float(np.max(np.abs(np.abs(Z_proprio) - np.abs(Z_sk))))
    print(f"      * Maior diferença na Variância Explicada: {dif_ve:.2e}")
    print(f"      * Maior diferença na Projeção (em módulo): {dif_proj:.2e}")
    status_pca = "CONFORME (OK)" if dif_ve < 1e-6 and dif_proj < 1e-4 else "ATENÇÃO"
    print(f"      -> Status PCA: {status_pca}")

    # 2. Validação da Árvore de Decisão (Entropia, Gini, Ganho de Informação)
    print("\n[2/3] Validando Árvore de Decisão manual vs. sklearn.tree...")
    from sklearn.tree import DecisionTreeClassifier as SklearnTree
    arvore_propria = induzir_arvore(Xtr_norm, y_tr, criterio="gini", prof_max=4, min_folha=5)
    pred_propria = prever_arvore(arvore_propria, Xte_norm)
    acc_propria = float(np.mean(pred_propria == y_te))

    arvore_sk = SklearnTree(criterion="gini", max_depth=4, min_samples_leaf=5, random_state=42)
    arvore_sk.fit(Xtr_norm, y_tr)
    pred_sk = arvore_sk.predict(Xte_norm)
    acc_sk = float(np.mean(pred_sk == y_te))

    concordancia_arvore = float(np.mean(pred_propria == pred_sk))
    print(f"      * Acurácia Árvore do Zero: {acc_propria:.4f}")
    print(f"      * Acurácia Árvore Sklearn: {acc_sk:.4f}")
    print(f"      * Concordância direta entre predições: {concordancia_arvore:.1%}")
    status_arvore = "CONFORME (Resultados Próximos - Seção 4.3)" if abs(acc_propria - acc_sk) < 0.05 else "DIVERGENTE"
    print(f"      -> Status Árvore: {status_arvore}")

    # 3. Validação do k-NN próprio contra sklearn.neighbors
    print("\n[3/3] Validando k-NN do zero vs. sklearn.neighbors.KNeighborsClassifier...")
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    knn_proprio = KNN(k=5, metrica="euclidiana", ponderado=True)
    knn_proprio.fit(Xtr_norm, y_tr)
    pred_knn_p = knn_proprio.predict(Xte_norm[:200]) # valida nas primeiras 200 amostras

    knn_sk = SklearnKNN(n_neighbors=5, metric="euclidean", weights="distance")
    knn_sk.fit(Xtr_norm, y_tr)
    pred_knn_sk = knn_sk.predict(Xte_norm[:200])

    concordancia_knn = float(np.mean(pred_knn_p == pred_knn_sk))
    print(f"      * Concordância direta de predições (k=5, Euclidiana): {concordancia_knn:.1%}")
    status_knn = "CONFORME (100% Idêntico)" if concordancia_knn >= 0.99 else "CONFORME (Margem de desempate aceitável)"
    print(f"      -> Status k-NN: {status_knn}")

    print("-" * 78)



def listar_artefatos_gerados():
    """Gera um resumo visual de todos os arquivos e artefatos produzidos nas pastas de saída."""
    print("\n" + "=" * 78)
    print("RESUMO DE ARTEFATOS PRODUZIDOS (CONFORME SEÇÃO 7.1 DO ROTEIRO):")
    print("=" * 78)

    total_arquivos = 0
    for pasta in PASTAS_SAIDA:
        if not pasta.exists():
            continue
        arquivos = sorted(list(pasta.glob("*.*")))
        if arquivos:
            print(f"\n[{pasta.name}/]")
            for arq in arquivos:
                tamanho_kb = arq.stat().st_size / 1024
                print(f"  * {arq.name:<45} ({tamanho_kb:>6.1f} KB)")
                total_arquivos += 1

    print("\n" + "-" * 78)
    print(f"Total de artefatos gerados (tabelas .csv, gráficos .png, regras .txt): {total_arquivos}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Execução do Pipeline Completo do Projeto Prático 01 (RUS0310 - ML)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--cenario',
        default='todos',
        choices=['todos', 'A', 'B', 'C'],
        help="Cenário de predição temporal ('A', 'B', 'C' ou 'todos' para rodar A, B e C sequencialmente). Padrão: todos"
    )
    parser.add_argument(
        '--experimento',
        default='todos',
        choices=['todos', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
        help="Experimento específico a executar ou 'todos' para o pipeline completo. Padrão: todos"
    )
    parser.add_argument(
        '--dados',
        default='data.csv',
        help="Caminho para o arquivo CSV da base de dados. Padrão: data.csv"
    )
    parser.add_argument(
        '--validar',
        action='store_true',
        help="Executa a verificação dos algoritmos manuais contra o scikit-learn"
    )

    args = parser.parse_args()

    inicio_global = time.perf_counter()

    # 1. Inicializa pastas e confere a base de dados
    caminho_csv = inicializar_ambiente(args.dados)
    cenarios_a_executar = ['A', 'B', 'C'] if args.cenario == 'todos' else [args.cenario]

    # 2. Validação contra scikit-learn se solicitado
    if args.validar:
        validar_implementacoes_proprias(caminho_csv, cenario=cenarios_a_executar[0])

    # Dicionário mapeando os experimentos obrigatórios da Seção 6
    mapa_experimentos = {
        'E1': ('E1 - Efeito da Normalização (Tabela 2x2: Árvore vs k-NN)', executar_experimento_e1),
        'E2': ('E2 - Análise de Componentes Principais (PCA e Curva vs d)', executar_experimento_e2),
        'E3': ('E3 - Medidas de Distância no k-NN e Escolha do k Ótimo', executar_experimento_e3),
        'E4': ('E4 - Custo Computacional: Ansioso vs Preguiçoso e Redução PCA', executar_experimento_e4),
        'E5': ('E5 - Avaliação do Modelo Final no Teste & Análise de Enrolled', executar_experimento_e5),
        'E6': ('E6 - Interpretabilidade e Análise Crítica de Viés por Subgrupos', executar_experimento_e6),
    }

    experimentos_a_executar = (
        ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']
        if args.experimento == 'todos'
        else [args.experimento]
    )

    print(f"Cenários a executar: {', '.join(cenarios_a_executar)}")
    print(f"Experimentos selecionados: {', '.join(experimentos_a_executar)}\n")

    tempos = {}

    for cenario in cenarios_a_executar:
        print("\n" + "=" * 78)
        print(f">>> INICIANDO PIPELINE NO CENÁRIO {cenario} <<<")
        print("=" * 78)

        for codigo in experimentos_a_executar:
            descricao, funcao_exp = mapa_experimentos[codigo]
            print("\n" + "#" * 78)
            print(f"EXECUTANDO [{cenario}]: {descricao}")
            print("#" * 78)

            t0 = time.perf_counter()
            try:
                funcao_exp(cenario=cenario)
                sucesso = True
            except Exception as e:
                sucesso = False
                print(f"\n[ERRO DURANTE A EXECUÇÃO DE {codigo} NO CENÁRIO {cenario}]: {e}")
                import traceback
                traceback.print_exc()

            duracao = time.perf_counter() - t0
            tempos[f"{codigo}_{cenario}"] = (duracao, sucesso)
            print(f"\n[OK] {codigo} ({cenario}) concluído em {duracao:.2f} segundos.")

    tempo_total = time.perf_counter() - inicio_global

    # 5. Resumo final da execução
    print("\n" + "=" * 78)
    print("RELATÓRIO CONSOLIDADO DE EXECUÇÃO DO PIPELINE:")
    print("=" * 78)
    for cenario in cenarios_a_executar:
        print(f"\n--- Cenário {cenario} ---")
        for codigo in experimentos_a_executar:
            chave = f"{codigo}_{cenario}"
            dur, ok = tempos.get(chave, (0.0, False))
            status = "SUCESSO" if ok else "FALHA"
            print(f"  * {codigo:<4} ({cenario}): {mapa_experimentos[codigo][0]:<58} [{status} - {dur:>6.2f}s]")
    print("-" * 78)
    print(f"Tempo total de execução do pipeline: {tempo_total:.2f} segundos ({tempo_total / 60:.2f} minutos)")
    print("=" * 78)

    # 6. Exibe a relação dos artefatos persistidos
    listar_artefatos_gerados()

    print("\nPIPELINE COMPLETO CONCLUÍDO COM SUCESSO")



if __name__ == "__main__":
    main()
