import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from dados import carregar_dados, separar_treino_teste, MAPA_DE_TIPOS
from preprocessamento import preparar
from arvoreDecisao import (entropia, indice_gini, ganho_informacao,
                           induzir_arvore, prever_arvore, podar_arvore,
                           extrair_regras, formatar_regras,
                           contar_folhas, profundidade_arvore, imprimir_arvore)

# configuração
SEED = 42
PASTA_SAIDA = 'saidas_arvore'
CRITERIO = 'gini'
MIN_FOLHA = 5
PROFUNDIDADES = list(range(1, 21))

# métricas
def matriz_confusao(y_real, y_pred, n_classes):
    """Linhas = classe real, colunas = classe prevista."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(y_real, y_pred):
        M[int(real), int(pred)] += 1
    return M

def relatorio_metricas(M):
    """Para a classe c: VP = M[c,c], FP = resto da coluna, FN = resto da linha."""
    precisao, revocacao, f1 = [], [], []
    for c in range(M.shape[0]):
        VP = M[c, c]
        FP = M[:, c].sum() - VP
        FN = M[c, :].sum() - VP
        P = VP / (VP + FP) if (VP + FP) > 0 else 0.0
        R = VP / (VP + FN) if (VP + FN) > 0 else 0.0
        F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        precisao.append(P)
        revocacao.append(R)
        f1.append(F)
    return {'acuracia': float(np.trace(M) / M.sum()),
            'precisao': np.array(precisao), 'revocacao': np.array(revocacao),
            'f1': np.array(f1), 'f1_macro': float(np.mean(f1)),
            'acuracia_balanceada': float(np.mean(revocacao))}

def f1_macro(y_real, y_pred, n_classes):
    """Métrica de decisão do projeto: média simples dos F1, peso 1/K por classe."""
    return relatorio_metricas(matriz_confusao(y_real, y_pred, n_classes))['f1_macro']

def auc_roc(y_real, scores):
    """AUC pela estatística de Mann-Whitney: soma das posições (ranks) dos
    positivos na lista ordenada por score."""
    y_real, scores = np.asarray(y_real), np.asarray(scores, dtype=float)
    n_pos, n_neg = int(np.sum(y_real == 1)), int(np.sum(y_real == 0))
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    ranks = np.empty(scores.size, dtype=float)
    ranks[np.argsort(scores, kind='mergesort')] = np.arange(1, scores.size + 1)
    for valor in np.unique(scores):
        iguais = (scores == valor)
        if iguais.sum() > 1:
            ranks[iguais] = ranks[iguais].mean()

    return float((ranks[y_real == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def score_binario(raiz, X):
    """Probabilidade da classe 1 na folha onde cada amostra cai, para a AUC.
    Com duas classes, acertos/n é a confiança da classe majoritária da folha."""
    scores = []
    for amostra in np.asarray(X, dtype=float):
        no = raiz
        while not no.folha():
            no = no.esq if amostra[no.atributo] <= no.limiar else no.dir
        confianca = no.acertos / no.n if no.n > 0 else 0.0
        scores.append(confianca if no.classe == 1 else 1.0 - confianca)
    return np.array(scores)

def validacao_cruzada(x, y, mapa, n_classes, prof, metodo=None):
    """CV estratificada 10-fold x 3 sobre o treino. A codificação e a
    normalização são reajustadas DENTRO de cada fold."""
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
    scores = []
    for idx_aj, idx_val in cv.split(x, y):
        a_tr, a_val, _ = preparar(x.iloc[idx_aj], x.iloc[idx_val], mapa, metodo=metodo)
        raiz = induzir_arvore(a_tr, y[idx_aj], CRITERIO, prof, MIN_FOLHA)
        scores.append(f1_macro(y[idx_val], prever_arvore(raiz, a_val), n_classes))
    return np.array(scores)

def salvar(linhas, nome, cenario, indice=False):
    df = linhas if isinstance(linhas, pd.DataFrame) else pd.DataFrame(linhas)
    df.to_csv(os.path.join(PASTA_SAIDA, f'{nome}_{cenario}.csv'), index=indice)

#  4.3.1 - Entropia, Gini e ganho conferidos contra o cálculo à mão
def exp_calculo_manual(cenario):
    aprovadas = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    desfecho = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])

    # ganho multi-ramo (ID3 original)
    def multiramo(criterio):
        medir = entropia if criterio == 'entropia' else indice_gini
        media = sum((desfecho[aprovadas == v].size / desfecho.size)
                    * medir(desfecho[aprovadas == v]) for v in np.unique(aprovadas))
        return medir(desfecho) - media

    linhas = [('H(S) da raiz', entropia(desfecho), 0.940),
              ('Gini(S) da raiz', indice_gini(desfecho), 0.459),
              ('Ganho(S, Aprovadas)', multiramo('entropia'), 0.246),
              ('Redução de Gini', multiramo('gini'), 0.116)]

    print('\n[1] Conferência contra o cálculo à mão (14 alunos do slide)')
    for nome, obtido, a_mao in linhas:
        print(f'    {nome:22s} código = {obtido:.4f} | à mão = {a_mao:.3f}')
    print('    Entropia dos ramos: ' + ', '.join(
        f'{n} = {entropia(desfecho[aprovadas == v]):.3f}'
        for v, n in [(0, 'Baixo'), (1, 'Médio'), (2, 'Alto')]))
    print(f'    Melhor corte binário (<= 1,5): {ganho_informacao(aprovadas, desfecho, 1.5):.3f}')

    salvar([{'grandeza': n, 'codigo': round(o, 4), 'a_mao': m} for n, o, m in linhas],
           'calculo_manual', cenario)

#  4.3.2 - Validação contra o DecisionTreeClassifier
def exp_validacao(X_tr, X_te, y_tr, y_te, n_classes, prof, cenario):
    raiz = induzir_arvore(X_tr, y_tr, CRITERIO, prof, MIN_FOLHA)
    pred = prever_arvore(raiz, X_te)

    ref = DecisionTreeClassifier(criterion=CRITERIO.replace('entropia', 'entropy'), max_depth=prof, min_samples_leaf=MIN_FOLHA, random_state=SEED).fit(X_tr, y_tr)
    pred_ref = ref.predict(X_te)

    f1_nossa = f1_macro(y_te, pred, n_classes)
    f1_ref = f1_macro(y_te, pred_ref, n_classes)
    acordo = float(np.mean(pred == pred_ref))

    print(f'\n[2] Validação contra o scikit-learn (prof_max = {prof})')
    print(f'    F1-macro própria = {f1_nossa:.4f} | sklearn = {f1_ref:.4f}')
    print(f'    Previsões idênticas: {acordo:.2%}')
    print(f'    Folhas: {contar_folhas(raiz)} | profundidade: {profundidade_arvore(raiz)}')

    salvar([{'prof_max': prof, 'f1_macro_nossa': round(f1_nossa, 4),
             'f1_macro_sklearn': round(f1_ref, 4), 'acordo': round(acordo, 4),
             'folhas': contar_folhas(raiz)}], 'validacao_sklearn', cenario)

#  4.3.3 - Efeito da profundidade: onde começa o sobreajuste
def exp_profundidade(x_tr, y_tr, mapa, n_classes, cenario):
    """Em cada fold cresce-se uma árvore até a profundidade máxima e avalia-se
    cada profundidade por truncamento: 1 ajuste por fold em vez de 20."""
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
    scores = {p: [] for p in PROFUNDIDADES}

    for idx_aj, idx_val in cv.split(x_tr, y_tr):
        a_tr, a_val, _ = preparar(x_tr.iloc[idx_aj], x_tr.iloc[idx_val], mapa)
        raiz = induzir_arvore(a_tr, y_tr[idx_aj], CRITERIO, max(PROFUNDIDADES), MIN_FOLHA)
        for p in PROFUNDIDADES:
            scores[p].append(f1_macro(y_tr[idx_val], prever_arvore(raiz, a_val, prof_corte=p), n_classes))

    X_tr, _, _ = preparar(x_tr, x_tr, mapa)
    raiz_tr = induzir_arvore(X_tr, y_tr, CRITERIO, max(PROFUNDIDADES), MIN_FOLHA)
    treino = [f1_macro(y_tr, prever_arvore(raiz_tr, X_tr, prof_corte=p), n_classes)
              for p in PROFUNDIDADES]
    media = [float(np.mean(scores[p])) for p in PROFUNDIDADES]
    desvio = [float(np.std(scores[p])) for p in PROFUNDIDADES]
    melhor = PROFUNDIDADES[int(np.argmax(media))]

    print('\n[3] Efeito da profundidade (10-fold x 3 sobre o treino)')
    print(f'    {"prof":>5s}{"treino":>10s}{"validação":>22s}{"vão":>9s}')
    for p, t, v, d in zip(PROFUNDIDADES, treino, media, desvio):
        print(f'    {p:>5d}{t:>10.4f}{f"{v:.4f} +- {d:.4f}":>22s}{t - v:>9.4f}')
    print(f'    Melhor profundidade: {melhor} (F1-macro = {max(media):.4f}); '
          f'o sobreajuste começa aí.')

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(PROFUNDIDADES, treino, 'o-', color='#D55E00', label='treino')
    plt.errorbar(PROFUNDIDADES, media, yerr=desvio, fmt='s--', color='#0072B2',
                 capsize=3, label='validação (10-fold x 3)')
    plt.axvline(melhor, ls=':', color='gray', label=f'melhor = {melhor}')
    plt.xlabel('Profundidade máxima')
    plt.ylabel('F1-macro')
    plt.title(f'Efeito da profundidade - cenário {cenario}')
    plt.xticks(PROFUNDIDADES[::2])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, f'profundidade_{cenario}.png'), dpi=150)

    salvar({'prof_max': PROFUNDIDADES, 'f1_treino': np.round(treino, 4),
            'f1_val_media': np.round(media, 4), 'f1_val_desvio': np.round(desvio, 4),
            'vao': np.round(np.array(treino) - np.array(media), 4)},
           'profundidade', cenario)
    return melhor


#  Poda por erro reduzido
def exp_poda(x_tr, x_te, y_tr, y_te, mapa, n_classes, cenario):
    # o conjunto de validação da poda sai do TREINO, nunca do teste
    x_aj, x_val, y_aj, y_val = train_test_split(
        x_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
    a_aj, a_val, _ = preparar(x_aj, x_val, mapa)
    _, a_te, _ = preparar(x_aj, x_te, mapa)

    raiz = induzir_arvore(a_aj, y_aj, CRITERIO, None, MIN_FOLHA)
    antes = (contar_folhas(raiz), f1_macro(y_te, prever_arvore(raiz, a_te), n_classes))

    podar_arvore(raiz, a_val, y_val)
    depois = (contar_folhas(raiz), f1_macro(y_te, prever_arvore(raiz, a_te), n_classes))

    print('\n[4] Poda por erro reduzido')
    print(f'    Antes  -> {antes[0]:4d} folhas | F1-macro no teste = {antes[1]:.4f}')
    print(f'    Depois -> {depois[0]:4d} folhas | F1-macro no teste = {depois[1]:.4f}')

    salvar([{'momento': 'antes', 'folhas': antes[0], 'f1_macro_teste': round(antes[1], 4)},
            {'momento': 'depois', 'folhas': depois[0], 'f1_macro_teste': round(depois[1], 4)}],
           'poda', cenario)


#  E5 - Modelo final no teste, e as regras SE-ENTÃO
def exp_modelo_final(X_tr, X_te, y_tr, y_te, nomes, classes, prof, cenario):
    raiz = induzir_arvore(X_tr, y_tr, CRITERIO, prof, MIN_FOLHA)
    M = matriz_confusao(y_te, prever_arvore(raiz, X_te), len(classes))
    r = relatorio_metricas(M)

    print(f'\n[5] E5 - Modelo final no teste (prof_max = {prof})')
    print('    ' + ' ' * 11 + ''.join(f'{c[:10]:>12s}' for c in classes))
    for c, nome in enumerate(classes):
        print(f'    {nome[:11]:<11s}' + ''.join(f'{v:12d}' for v in M[c]))
    print(f'    Acurácia = {r["acuracia"]:.4f} | F1-macro = {r["f1_macro"]:.4f} '
          f'| balanceada = {r["acuracia_balanceada"]:.4f}')
    print(f'    {"classe":<12s}{"precisão":>11s}{"revocação":>11s}{"f1":>9s}{"suporte":>10s}')
    for c, nome in enumerate(classes):
        print(f'    {nome[:12]:<12s}{r["precisao"][c]:>11.4f}{r["revocacao"][c]:>11.4f}'
              f'{r["f1"][c]:>9.4f}{M[c].sum():>10d}')

    i = list(classes).index('Enrolled')
    print(f'    Dos {M[i].sum()} Enrolled reais: ' + ', '.join(
        f'{classes[c]} = {M[i, c]} ({M[i, c] / M[i].sum():.0%})' for c in range(len(classes))))

    salvar([{'classe': n, 'precisao': round(r['precisao'][c], 4),
             'revocacao': round(r['revocacao'][c], 4), 'f1': round(r['f1'][c], 4),
             'suporte': int(M[c].sum())} for c, n in enumerate(classes)]
           + [{'classe': 'MACRO', 'precisao': '', 'revocacao': '',
               'f1': round(r['f1_macro'], 4), 'suporte': int(M.sum())}],
           'teste_final', cenario)
    salvar(pd.DataFrame(M, index=classes, columns=classes), 'matriz_confusao',
           cenario, indice=True)

    rasa = induzir_arvore(X_tr, y_tr, CRITERIO, 3, 20)
    regras = extrair_regras(rasa, nomes)
    for reg in regras:
        reg['classe'] = classes[reg['classe']]

    print('\n[6] Regras SE-ENTÃO (árvore rasa, prof_max = 3)')
    print(imprimir_arvore(rasa, nomes, list(classes)))
    print(formatar_regras(regras, top=4))

    with open(os.path.join(PASTA_SAIDA, f'regras_{cenario}.txt'), 'w', encoding='utf-8') as f:
        f.write(imprimir_arvore(rasa, nomes, list(classes)) + '\n' + formatar_regras(regras))
    salvar([{'regra': ' E '.join(g['condicoes']), 'classe': g['classe'], 'n': g['n'],
             'cobertura': round(g['cobertura'], 4), 'confianca': round(g['confianca'], 4)}
            for g in sorted(regras, key=lambda z: -z['cobertura'])], 'regras', cenario)
    return raiz


#  E1 - Efeito da normalização (metade da árvore na tabela 2x2)
def exp_normalizacao(x_tr, y_tr, mapa, n_classes, prof, cenario):
    print('\n[7] E1 - Efeito da normalização')
    linhas = []
    for metodo, rotulo in [(None, 'sem'), ('minmax', 'min-max'), ('zscore', 'z-score')]:
        s = validacao_cruzada(x_tr, y_tr, mapa, n_classes, prof, metodo=metodo)
        print(f'    {rotulo:>10s}: {s.mean():.4f} +- {s.std():.4f}')
        linhas.append({'normalizacao': rotulo, 'f1_macro_media': round(float(s.mean()), 4),
                       'f1_macro_desvio': round(float(s.std()), 4)})
    salvar(linhas, 'e1_normalizacao', cenario)


#  E4 - Custo computacional
def exp_custo(X_tr, X_te, y_tr, prof, cenario):
    t0 = time.perf_counter()
    raiz = induzir_arvore(X_tr, y_tr, CRITERIO, prof, MIN_FOLHA)
    t_treino = time.perf_counter() - t0

    # replica as instâncias de teste até chegar a 1.000, como o PP01 pede
    X_mil = np.tile(X_te, (int(np.ceil(1000 / len(X_te))), 1))[:1000]
    t0 = time.perf_counter()
    prever_arvore(raiz, X_mil)
    t_predicao = time.perf_counter() - t0

    print('\n[8] E4 - Custo computacional')
    print(f'    Treino ({len(X_tr)} instâncias): {1000 * t_treino:.1f} ms')
    print(f'    Predição de 1.000 instâncias: {1000 * t_predicao:.1f} ms '
          f'({t_treino / t_predicao:.0f}x mais barato)')

    salvar([{'modelo': f'arvore (prof_max={prof})',
             'tempo_treino_ms': round(1000 * t_treino, 2),
             'tempo_predicao_1000_ms': round(1000 * t_predicao, 2)}], 'e4_custo', cenario)


#  Análise binária secundária (Dropout x demais), com AUC-ROC
def exp_binario(X_tr, X_te, y_tr, y_te, classes, prof, cenario):
    i = list(classes).index('Dropout')
    y_tr_bin, y_te_bin = (y_tr == i).astype(int), (y_te == i).astype(int)

    raiz = induzir_arvore(X_tr, y_tr_bin, CRITERIO, prof, MIN_FOLHA)
    r = relatorio_metricas(matriz_confusao(y_te_bin, prever_arvore(raiz, X_te), 2))
    auc = auc_roc(y_te_bin, score_binario(raiz, X_te))

    print('\n[9] Análise binária secundária: Dropout x demais')
    print(f'    Acurácia = {r["acuracia"]:.4f} | F1-macro = {r["f1_macro"]:.4f} '
          f'| F1 do Dropout = {r["f1"][1]:.4f} | AUC-ROC = {auc:.4f}')

    salvar([{'acuracia': round(r['acuracia'], 4), 'f1_macro': round(r['f1_macro'], 4),
             'f1_dropout': round(r['f1'][1], 4), 'auc_roc': round(auc, 4)}],
           'binario', cenario)


#  4.6 - Taxa de erro por subgrupo
def exp_vies(raiz, x_te, X_te, y_te, classes, cenario):
    pred = prever_arvore(raiz, X_te)
    acertou = (pred == y_te)

    subgrupos = {'Gender': {0: 'feminino', 1: 'masculino'},
                 'Scholarship holder': {0: 'sem bolsa', 1: 'bolsista'},
                 'Debtor': {0: 'sem débito', 1: 'devedor'}}

    print('\n[10] Taxa de erro por subgrupo')
    linhas = []
    for coluna, rotulos in subgrupos.items():
        if coluna not in x_te.columns:
            continue
        taxas = []
        for valor, rotulo in rotulos.items():
            m = (x_te[coluna].values == valor)
            if m.sum() == 0:
                continue
            erro = float(1 - acertou[m].mean())
            taxas.append(erro)
            print(f'    {coluna:<20s}{rotulo:<13s} n={m.sum():>4d}  erro={erro:.4f}  '
                  f'F1-macro={f1_macro(y_te[m], pred[m], len(classes)):.4f}')
            linhas.append({'atributo': coluna, 'subgrupo': rotulo, 'n': int(m.sum()),
                           'taxa_erro': round(erro, 4),
                           'f1_macro': round(f1_macro(y_te[m], pred[m], len(classes)), 4)})
        if len(taxas) == 2:
            print(f'    {"":<20s}{"diferença":<13s}      erro={abs(taxas[0] - taxas[1]):.4f}')

    salvar(linhas, 'vies', cenario)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dados', default='data.csv')
    ap.add_argument('--cenario', default='C', choices=['A', 'B', 'C'])
    ap.add_argument('--prof', type=int, default=None,
                    help='profundidade máxima; o padrão usa a escolhida pela validação')
    ap.add_argument('--mostrar', action='store_true', help='abrir os gráficos na tela')
    args = ap.parse_args()

    if not args.mostrar:
        plt.switch_backend('Agg')
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    cen = args.cenario

    x, y = carregar_dados(args.dados, cen)
    classes, y_num = np.unique(y, return_inverse=True)
    x_tr, x_te, y_tr, y_te = separar_treino_teste(x, y_num, seed=SEED)
    mapa = {c: MAPA_DE_TIPOS[c] for c in x_tr.columns}
    X_tr, X_te, nomes = preparar(x_tr, x_te, mapa)

    print(f'Cenário {cen}: {x.shape[0]} alunos, {x.shape[1]} atributos')
    print('Classes: ' + ', '.join(f'{i} = {c}' for i, c in enumerate(classes)))
    print(f'Treino: {X_tr.shape} | teste: {X_te.shape} | após codificar: {len(nomes)}')

    exp_calculo_manual(cen)
    melhor = exp_profundidade(x_tr, y_tr, mapa, len(classes), cen)
    prof = args.prof if args.prof is not None else melhor

    exp_validacao(X_tr, X_te, y_tr, y_te, len(classes), prof, cen)
    exp_poda(x_tr, x_te, y_tr, y_te, mapa, len(classes), cen)
    raiz = exp_modelo_final(X_tr, X_te, y_tr, y_te, nomes, classes, prof, cen)
    exp_normalizacao(x_tr, y_tr, mapa, len(classes), prof, cen)
    exp_custo(X_tr, X_te, y_tr, prof, cen)
    exp_binario(X_tr, X_te, y_tr, y_te, classes, prof, cen)
    exp_vies(raiz, x_te, X_te, y_te, classes, cen)

    print(f'\nArquivos salvos em "{PASTA_SAIDA}/".')
    if args.mostrar:
        plt.show()


if __name__ == '__main__':
    main()