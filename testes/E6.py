# Experimento E6: Interpretabilidade e Análise Crítica de Viés
# - Análise de Interpretabilidade (Árvore de Decisão vs. PCA)
# - Regras de Decisão SE-ENTÃO extraídas do modelo para a coordenação
# - Taxa de erro e F1-macro por subgrupos sensíveis (Gênero, Bolsista, Devedor)
# - Discussão ética: Profecia Autorrealizável e uso de dados socioeconômicos

import sys
from pathlib import Path
import os

DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text

from dados import carregar_dados, separar_treino_teste, MAPA_DE_TIPOS
from preprocessamento import preparar, construir_atributos
from arvoreDecisao_teste import matriz_confusao, relatorio_metricas

# Configurações
plt.switch_backend('Agg')
SEMENTE = 42
CENARIO = 'C'
PROF_OTIMA = 6
MIN_FOLHA = 5
PASTA_SAIDA = os.path.join(str(DIRETORIO_RAIZ), 'saidas_experimentos')
os.makedirs(PASTA_SAIDA, exist_ok=True)

def executar_experimento_e6():
    print("=" * 76)
    print(f"EXPERIMENTO E6: INTERPRETABILIDADE E ANÁLISE DE VIÉS (CENÁRIO {CENARIO})")
    print("=" * 76)

    # 1. Carregamento dos dados
    caminho_csv = str(DIRETORIO_RAIZ / "data.csv")
    X, y = carregar_dados(caminho_csv, cenario=CENARIO)
    classes, y_numerico = np.unique(y, return_inverse=True)

    X_treino_raw, X_teste_raw, y_treino, y_teste = separar_treino_teste(
        X, y_numerico, seed=SEMENTE
    )

    X_treino = construir_atributos(X_treino_raw)
    X_teste = construir_atributos(X_teste_raw)

    mapa_tipos = {c: MAPA_DE_TIPOS.get(c, 'continuo') for c in X_treino.columns}

    # Pré-processamento
    X_tr_proc, X_te_proc, nomes_atributos = preparar(X_treino, X_teste, mapa_tipos, metodo='minmax')

    # 2. Treinamento do Modelo Campeão (Árvore de Decisão)
    modelo = DecisionTreeClassifier(
        criterion='gini',
        max_depth=PROF_OTIMA,
        min_samples_leaf=MIN_FOLHA,
        random_state=SEMENTE
    )
    modelo.fit(X_tr_proc, y_treino)
    y_pred = modelo.predict(X_te_proc)

    # =========================================================================
    # PARTE 1: INTERPRETABILIDADE PARA A COORDENAÇÃO (REGRAS SE-ENTÃO)
    # =========================================================================
    print("\n[1] Interpretabilidade do Modelo para a Coordenação Acadêmica:")
    print("    Extraindo as principais regras de decisão nos primeiros níveis da árvore:")
    
    regras_arvore = export_text(
        modelo,
        feature_names=nomes_atributos,
        max_depth=3,
        show_weights=True
    )

    print("-" * 76)
    linhas_exibicao = regras_arvore.split("\n")[:18]
    for l in linhas_exibicao:
        print("    " + l)
    print("    [... demais ramificações e folhas na pasta saidas_experimentos/]")
    print("-" * 76)

    caminho_regras = os.path.join(PASTA_SAIDA, f'e6_regras_decisao_{CENARIO}.txt')
    with open(caminho_regras, 'w', encoding='utf-8') as f:
        f.write(regras_arvore)
    print(f"    -> Regras completas salvas em: {caminho_regras}")

    # =========================================================================
    # PARTE 2: TAXA DE ERRO POR SUBGRUPO SENSÍVEL (SEÇÃO 4.6 DO ROTEIRO)
    # =========================================================================
    print("\n[2] Análise Quantitativa de Viés por Subgrupos Sensíveis (Conjunto de Teste):")
    
    acertos = (y_pred == y_teste)

    # Subgrupos obrigatórios do roteiro (Gênero, Bolsista, Devedor)
    definicao_subgrupos = {
        'Gender': {
            'nome_legivel': 'Gênero',
            'valores': {0: 'Feminino', 1: 'Masculino'}
        },
        'Scholarship holder': {
            'nome_legivel': 'Condição de Bolsista',
            'valores': {0: 'Não Bolsista', 1: 'Bolsista'}
        },
        'Debtor': {
            'nome_legivel': 'Situação Financeira',
            'valores': {0: 'Sem Débito', 1: 'Devedor'}
        }
    }

    tabela_vies = []

    for col, info in definicao_subgrupos.items():
        if col not in X_teste_raw.columns:
            continue
        
        print(f"\n-> Subgrupo: {info['nome_legivel']} ({col})")
        taxas_erro_grupo = []

        for val, rotulo in info['valores'].items():
            mascara = (X_teste_raw[col].values == val)
            n_amostras = int(mascara.sum())
            if n_amostras == 0:
                continue

            erro = float(1.0 - acertos[mascara].mean())
            taxas_erro_grupo.append(erro)

            # F1-macro no subgrupo
            M_sub = matriz_confusao(y_teste[mascara], y_pred[mascara], len(classes))
            f1_sub = relatorio_metricas(M_sub)['f1_macro']

            print(f"   * {rotulo:<14} | N = {n_amostras:>4d} | Taxa de Erro = {erro*100:>6.2f}% | F1-Macro = {f1_sub:.4f}")

            tabela_vies.append({
                'atributo': col,
                'subgrupo': rotulo,
                'n_amostras': n_amostras,
                'taxa_erro': round(erro, 4),
                'acuracia': round(1.0 - erro, 4),
                'f1_macro': round(f1_sub, 4)
            })

        if len(taxas_erro_grupo) == 2:
            disparidade = abs(taxas_erro_grupo[0] - taxas_erro_grupo[1]) * 100
            print(f"   >> Disparidade absoluta de erro entre os grupos: {disparidade:.2f} pontos percentuais")

    # Salvar tabela de viés em CSV
    df_vies = pd.DataFrame(tabela_vies)
    caminho_csv_vies = os.path.join(PASTA_SAIDA, f'e6_analise_vies_{CENARIO}.csv')
    df_vies.to_csv(caminho_csv_vies, index=False)
    print(f"\n-> Tabela de viés salva em: {caminho_csv_vies}")

    # =========================================================================
    # PARTE 3: GRÁFICO COMPARATIVO DE TAXA DE ERRO POR SUBGRUPO
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    rotulos_grafico = [f"{r['atributo']}\n({r['subgrupo']})" for r in tabela_vies]
    erros_grafico = [r['taxa_erro'] * 100 for r in tabela_vies]
    cores = ['#0072B2' if 'Feminino' in r['subgrupo'] or 'Bolsista' in r['subgrupo'] or 'Sem Débito' in r['subgrupo']
             else '#D55E00' for r in tabela_vies]

    barras = ax.bar(range(len(erros_grafico)), erros_grafico, color=cores, width=0.55)
    ax.set_xticks(range(len(rotulos_grafico)))
    ax.set_xticklabels(rotulos_grafico, fontsize=9)
    ax.set_ylabel('Taxa de Erro (%)')
    ax.set_title(f'Taxa de Erro por Subgrupo Sensível - Modelo Final (Cenário {CENARIO})')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(erros_grafico) + 8)

    for bar in barras:
        altura = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, altura + 0.8, f'{altura:.1f}%', ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    caminho_fig_vies = os.path.join(PASTA_SAIDA, f'e6_taxa_erro_subgrupos_{CENARIO}.png')
    fig.savefig(caminho_fig_vies, dpi=150)
    plt.close(fig)
    print(f"-> Gráfico de viés salvo em: {caminho_fig_vies}")

    # =========================================================================
    # PARTE 4: DISCUSSÃO TEÓRICA E ÉTICA CONSOLIDADA (PARA O RELATÓRIO)
    # =========================================================================
    print("\n" + "=" * 76)
    print("SÍNTESE DOS PONTOS CRÍTICOS EXIGIDOS NO RELATÓRIO TÉCNICO (E6):")
    print("=" * 76)
    print("1. O Custo do PCA para a Coordenação (Interpretabilidade vs. Acurácia):")
    print("   * A Árvore de Decisão oferece regras transparentes e acionáveis, como:")
    print("     'SE aprovadas_2sem <= 2 E mensalidade_em_dia = Não -> Evasão'.")
    print("     Isso permite que a coordenação oriente tutoria pedagógica ou auxílio financeiro.")
    print("   * No PCA, as variáveis viram combinações lineares abstratas (Z1, Z2...).")
    print("     O modelo vira caixa-preta: o coordenador sabe que o aluno pode evadir, mas")
    print("     não sabe o motivo nem como intervir. O ganho marginal do PCA não compensa")
    print("     a perda da capacidade de intervenção.")
    print("\n2. Risco de Profecia Autorrealizável (Self-Fulfilling Prophecy):")
    print("   * Rotular precocemente um aluno como 'provável evadido' pode alterar a forma")
    print("     como professores e a própria instituição o tratam (menores expectativas,")
    print("     menos oportunidades ou isolamento). Esse tratamento desmotivador pode induzir")
    print("     a evasão que o modelo previa.")
    print("\n3. Dilema Ético do Uso de Atributos Socioeconômicos e Familiares:")
    print("   * Atributos como escolaridade/profissão dos pais e situação financeira carregam")
    print("     sinal preditivo, mas penalizar ou classificar o estudante por fatores fora")
    print("     de seu controle é eticamente questionável. O modelo deve priorizar o rendimento")
    print("     acadêmico e a assiduidade recente como base para ações de suporte.")
    print("=" * 76)

if __name__ == "__main__":
    executar_experimento_e6()
