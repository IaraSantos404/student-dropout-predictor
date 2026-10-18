# Experimento E5: Avaliação do Modelo Final no Conjunto de Teste
# - Modelo escolhido pela validação cruzada: Árvore de Decisão (prof_max=6, min_samples_leaf=5)
# - Avaliação única no conjunto de teste independente (20% dos dados)
# - Matriz de Confusão e Métricas por Classe (Precisão, Revocação, F1-Score)
# - Análise detalhada dos erros da classe "Enrolled"

import sys
from pathlib import Path
import os

DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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

def executar_experimento_e5():
    print("=" * 76)
    print(f"EXPERIMENTO E5: AVALIAÇÃO DO MODELO FINAL NO TESTE (CENÁRIO {CENARIO})")
    print("=" * 76)

    # 1. Carregamento dos dados e partição Treino/Teste (80% / 20%)
    caminho_csv = str(DIRETORIO_RAIZ / "data.csv")
    X, y = carregar_dados(caminho_csv, cenario=CENARIO)
    classes, y_numerico = np.unique(y, return_inverse=True)

    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(
        X, y_numerico, seed=SEMENTE
    )

    X_treino = construir_atributos(X_treino)
    X_teste = construir_atributos(X_teste)

    mapa_tipos = {c: MAPA_DE_TIPOS.get(c, 'continuo') for c in X_treino.columns}

    # Pré-processamento ajustado no treino e aplicado no teste
    X_tr_proc, X_te_proc, nomes_atributos = preparar(X_treino, X_teste, mapa_tipos, metodo='minmax')

    print(f"Dados prontos:")
    print(f"  * Conjunto de Treino: {len(X_tr_proc)} instâncias")
    print(f"  * Conjunto de Teste:  {len(X_te_proc)} instâncias (avaliado UMA ÚNICA VEZ)")
    print(f"  * Classes do problema: {list(classes)}")

    # 2. Ajuste do modelo campeão da validação cruzada
    print(f"\n[1] Treinando o Modelo Final Escolhido:")
    print(f"    -> Árvore de Decisão (Critério=Gini, prof_max={PROF_OTIMA}, min_samples_leaf={MIN_FOLHA})")
    modelo = DecisionTreeClassifier(
        criterion='gini',
        max_depth=PROF_OTIMA,
        min_samples_leaf=MIN_FOLHA,
        random_state=SEMENTE
    )
    modelo.fit(X_tr_proc, y_treino)

    # 3. Predição no conjunto de teste independente
    print("[2] Avaliando previsões no conjunto de teste...")
    y_pred = modelo.predict(X_te_proc)

    # 4. Matriz de Confusão e Métricas
    M = matriz_confusao(y_teste, y_pred, len(classes))
    relatorio = relatorio_metricas(M)

    # Exibição da Matriz de Confusão
    print("\n" + "=" * 76)
    print("MATRIZ DE CONFUSÃO NO CONJUNTO DE TESTE:")
    print("=" * 76)
    header = f"{'Classe Real':<14} | " + " | ".join(f"{c[:10]:>10}" for c in classes) + " | Total"
    print(header)
    print("-" * len(header))
    for i, c_real in enumerate(classes):
        linha = f"{c_real:<14} | " + " | ".join(f"{M[i, j]:>10d}" for j in range(len(classes))) + f" | {M[i].sum():>5d}"
        print(linha)
    print("-" * len(header))

    # Exibição do Relatório de Métricas por Classe
    print("\n" + "=" * 76)
    print("MÉTRICAS DETALHADAS POR CLASSE (CONJUNTO DE TESTE):")
    print("=" * 76)
    print(f"{'Classe':<14} | {'Precisão':<10} | {'Revocação':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-" * 65)
    for i, c in enumerate(classes):
        p = relatorio['precisao'][i]
        r = relatorio['revocacao'][i]
        f = relatorio['f1'][i]
        sup = M[i].sum()
        print(f"{c:<14} | {p:>9.4f}  | {r:>9.4f}  | {f:>9.4f}  | {sup:>7d}")
    print("-" * 65)
    print(f"{'MÉDIA MACRO':<14} | {np.mean(relatorio['precisao']):>9.4f}  | {np.mean(relatorio['revocacao']):>9.4f}  | {relatorio['f1_macro']:>9.4f}  | {M.sum():>7d}")
    print(f"\nAcurácia Global:      {relatorio['acuracia']:.4f} ({relatorio['acuracia']*100:.2f}%)")
    print(f"Acurácia Balanceada:  {relatorio['acuracia_balanceada']:.4f}")
    print(f"F1-Macro (Decisão):   {relatorio['f1_macro']:.4f}")

    # =========================================================================
    # 5. DIAGNÓSTICO E ANÁLISE DETALHADA DA CLASSE "ENROLLED"
    # =========================================================================
    idx_enrolled = list(classes).index('Enrolled')
    total_enrolled = M[idx_enrolled].sum()
    acertos_enrolled = M[idx_enrolled, idx_enrolled]
    para_dropout = M[idx_enrolled, list(classes).index('Dropout')]
    para_graduate = M[idx_enrolled, list(classes).index('Graduate')]

    print("\n" + "=" * 76)
    print("DIAGNÓSTICO CRÍTICO DA CLASSE 'ENROLLED':")
    print("=" * 76)
    print(f"Total de alunos 'Enrolled' reais no teste: {total_enrolled}")
    print(f"  * Acertados como Enrolled:   {acertos_enrolled:3d} ({acertos_enrolled / total_enrolled:.1%})")
    print(f"  * Confundidos com Dropout:    {para_dropout:3d} ({para_dropout / total_enrolled:.1%})")
    print(f"  * Confundidos com Graduate:   {para_graduate:3d} ({para_graduate / total_enrolled:.1%})")
    print(f"  * Taxa de erro para Enrolled: {(total_enrolled - acertos_enrolled) / total_enrolled:.1%}")
    print("\nPOR QUE A CLASSE 'ENROLLED' É AMBÍGUA E TEM DESEMPENHO INFERIOR?")
    print("1. Natureza Transitória: 'Enrolled' não é um desfecho definitivo, mas sim um estado")
    print("   temporário de quem ainda está cursando. Alguns alunos matriculados estão com ótimo")
    print("   desempenho (perfil idêntico a Graduate), enquanto outros acumulam reprovações")
    print("   (perfil idêntico a Dropout).")
    print("2. Desbalanceamento: É a classe minoritária da base (~17.9%), o que faz os modelos")
    print("   terem menos exemplos de treino e tenderem a prever as classes majoritárias.")
    print("=" * 76)

    # 6. Salvar Matriz de Confusão em Gráfico Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cax = ax.matshow(M, cmap='Blues')
    fig.colorbar(cax)

    for i in range(len(classes)):
        for j in range(len(classes)):
            cor_texto = "white" if M[i, j] > M.max() / 2 else "black"
            ax.text(j, i, f"{M[i, j]}", ha='center', va='center', color=cor_texto, fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=20)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Classe Prevista', fontweight='bold', labelpad=10)
    ax.set_ylabel('Classe Real', fontweight='bold')
    ax.set_title(f'Matriz de Confusão - Modelo Final (Cenário {CENARIO})', pad=15)
    fig.tight_layout()

    caminho_fig = os.path.join(PASTA_SAIDA, f'e5_matriz_confusao_{CENARIO}.png')
    fig.savefig(caminho_fig, dpi=150)
    plt.close(fig)
    print(f"\n-> Gráfico da Matriz de Confusão salvo em: {caminho_fig}")

    # Salvar métricas em CSV
    df_metricas = pd.DataFrame({
        'classe': list(classes) + ['MÉDIA MACRO'],
        'precisao': list(np.round(relatorio['precisao'], 4)) + [round(float(np.mean(relatorio['precisao'])), 4)],
        'revocacao': list(np.round(relatorio['revocacao'], 4)) + [round(float(np.mean(relatorio['revocacao'])), 4)],
        'f1': list(np.round(relatorio['f1'], 4)) + [round(float(relatorio['f1_macro']), 4)],
        'suporte': [int(M[c].sum()) for c in range(len(classes))] + [int(M.sum())]
    })
    caminho_csv = os.path.join(PASTA_SAIDA, f'e5_metricas_teste_{CENARIO}.csv')
    df_metricas.to_csv(caminho_csv, index=False)
    print(f"-> Métricas salvas em CSV: {caminho_csv}")

if __name__ == "__main__":
    executar_experimento_e5()
