import argparse
import os
 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
# AJUSTE o nome do módulo abaixo para o arquivo onde estão carregar_dados,
# separar_treino_teste e MAPA_DE_TIPOS.
from dados import carregar_dados, separar_treino_teste, MAPA_DE_TIPOS
from pca import pca_ajustar, pca_projetar, cargas_componentes
 
# Configuração
SEED = 42
PASTA_SAIDA = 'saidas_pca'
ORDINAIS_NO_BLOCO_NUMERICO = True
REGIMES = {'Com padronização': True, 'Sem padronização': False}
CORES_CLASSES = {'Dropout': '#D55E00', 'Enrolled': '#0072B2', 'Graduate': '#009E73'}
 
 
# Preparação dos dados
def verificar_escala_das_notas(x):
    """Avisa se alguma nota de unidade curricular passa de 20 (provável erro de leitura)."""
    for c in x.columns:
        if c.startswith('Curricular') and 'grade' in c.lower():
            n = int((x[c] > 20).sum())
            if n:
                print(f'  ATENÇÃO: {n} valor(es) acima de 20 em "{c}" '
                      f'(máximo = {x[c].max():g}). Confira o CSV: pode ser erro de '
                      f'formatação decimal e distorce o PCA.')
 
 
def dividir_blocos(colunas):
    """Separa as colunas em bloco numérico (vai para o PCA) e bloco categórico."""
    tipos_num = {'continuo'} | ({'ordinal'} if ORDINAIS_NO_BLOCO_NUMERICO else set())
    num = [c for c in colunas if MAPA_DE_TIPOS[c] in tipos_num]
    cat = [c for c in colunas if c not in num]
    return num, cat
 
 
def preparar_numericas(n_tr, n_te, padronizar):
    """Padroniza (média 0, desvio 1) usando só o treino, ou devolve os valores crus."""
    if padronizar:
        sc = StandardScaler().fit(n_tr)
        return sc.transform(n_tr), sc.transform(n_te)
    return n_tr.to_numpy(dtype=float), n_te.to_numpy(dtype=float)
 
 
def codificar_categoricas(c_tr, c_te):
    """One-hot ajustado no treino. Colunas já binárias ficam com uma coluna só."""
    if c_tr.shape[1] == 0:
        return np.zeros((len(c_tr), 0)), np.zeros((len(c_te), 0)), []
    enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)
    a = enc.fit_transform(c_tr)
    b = enc.transform(c_te)
    return a, b, list(enc.get_feature_names_out(c_tr.columns))
 
 
def escolher_d(ve, alvo):
    """Menor número de componentes cuja variância acumulada atinge o alvo."""
    cum = np.cumsum(ve)
    return int(np.argmax(cum >= alvo - 1e-12) + 1)
 
 
#  Implementação a partir da covariância, validada contra o scikit-learn
def exp_validacao(X):
    media, W, ve = pca_ajustar(X, None)
    Z = pca_projetar(X, media, W)
    ref = PCA().fit(X)
    Zr = ref.transform(X)
    dif_ve = np.max(np.abs(ve - ref.explained_variance_ratio_))
    dif_z = np.max(np.abs(np.abs(Z) - np.abs(Zr)))
    print('\n[1] Validação do PCA manual contra o scikit-learn (dados padronizados)')
    print(f'    maior diferença na variância explicada: {dif_ve:.2e}')
    print(f'    maior diferença na projeção (em módulo, por causa do sinal): {dif_z:.2e}')
    print('    =>', 'OK, bate com o sklearn' if dif_ve < 1e-8 and dif_z < 1e-6
          else 'CONFERIR: diferença acima da tolerância')
 
 

#  Efeito da padronização: variâncias brutas e cargas

def exp_padronizacao(x_tr_num, dados_num, nomes_num, pasta):
    print('\n[2] Efeito da padronização')
    var = x_tr_num.var()
    frac = (var / var.sum()).sort_values(ascending=False)
    print('    Variância bruta: atributos que mais pesam na variância total (sem padronizar):')
    for nome, f in frac.head(5).items():
        print(f'      {nome}: {f:.1%} da variância total ({var[nome]:.4g})')
    print(f'      os 3 primeiros somam {frac.head(3).sum():.1%}; '
          f'os 5 menores somam {frac.tail(5).sum():.4%}')
 
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    print('    Cargas das 3 primeiras componentes (5 atributos de maior peso em módulo):')
    for i, (nome, (n_tr, _)) in enumerate(dados_num.items()):
        media, W, ve = pca_ajustar(n_tr, 3)
        car = cargas_componentes(W, nomes_num, 3)
        sufixo = 'com' if REGIMES[nome] else 'sem'
        car.to_csv(os.path.join(pasta, f'e2_cargas_{sufixo}_padronizacao.csv'))
        print(f'    --- {nome} ---')
        for j, pc in enumerate(car.columns):
            top = car[pc].reindex(car[pc].abs().sort_values(ascending=False).index).head(6)
            print(f'    {pc} ({ve[j]:.1%}): ' +
                  '; '.join(f'{a} ({v:+.2f})' for a, v in top.head(5).items()))
            ax = axs[i, j]
            ax.barh(range(len(top)), top.values[::-1],
                    color=['#D55E00' if v < 0 else '#0072B2' for v in top.values[::-1]])
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([a if len(a) < 38 else a[:35] + '...' for a in top.index[::-1]],
                               fontsize=8)
            ax.set_title(f'{nome} - {pc} ({ve[j]:.1%})', fontsize=10)
            ax.axvline(0, color='black', lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, 'e2_cargas.png'), dpi=150)
 
 

#  Variância explicada e escolha de d
def exp_variancia(ves, alvo, caminho):
    ds = {nome: escolher_d(ve, alvo) for nome, ve in ves.items()}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (nome, ve) in zip(axs, ves.items()):
        x = np.arange(1, len(ve) + 1)
        ax.bar(x, ve, color='#9ecae1', label='Individual')
        ax.plot(x, np.cumsum(ve), 'o-', color='#08519c', ms=4, label='Acumulada')
        ax.axhline(alvo, ls='--', color='gray', lw=1)
        ax.axvline(ds[nome], ls=':', color='#D55E00', lw=1.5)
        ax.text(ds[nome] + 0.3, 0.05, f'd = {ds[nome]}', color='#D55E00')
        ax.set_title(f'{nome}\n(critério: {alvo:.0%} de variância acumulada)')
        ax.set_xlabel('Componente principal')
        ax.set_ylim(0, 1.02)
        if len(ve) <= 25:
            ax.set_xticks(x)
        ax.legend(loc='center right')
    axs[0].set_ylabel('Variância explicada')
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
 
    print(f'\n[3] Variância explicada (critério declarado: {alvo:.0%} acumulado)')
    for nome, ve in ves.items():
        print(f'    {nome}: PC1 = {ve[0]:.1%}, PC1+PC2 = {ve[:2].sum():.1%}, '
              f'd(90%) = {escolher_d(ve, 0.90)}, d(95%) = {escolher_d(ve, 0.95)}, '
              f'total de componentes = {len(ve)}')
    return ds
 
 
#  Projeção 2D colorida por classe
def recall_centroide(Z, y):
    """Recall por classe de um classificador de centroide mais próximo em 2D."""
    classes = np.unique(y)
    cent = np.array([Z[y == c].mean(axis=0) for c in classes])
    dist = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1)
    pred = classes[dist.argmin(axis=1)]
    return {c: float((pred[y == c] == c).mean()) for c in classes}
 
 
def exp_projecao_2d(dados_num, y_tr, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    classes = list(np.unique(y_tr))
    ordem = [c for c in classes if c != 'Enrolled'] + (['Enrolled'] if 'Enrolled' in classes else [])
    print('\n[4] Projeção 2D (treino): recall por classe com centroide mais próximo em 2D')
    for ax, (nome, (n_tr, _)) in zip(axs, dados_num.items()):
        media, W, ve = pca_ajustar(n_tr, 2)
        Z = pca_projetar(n_tr, media, W)
        for c in ordem:
            m = y_tr == c
            ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.45,
                       color=CORES_CLASSES.get(c, 'gray'), label=c)
        ax.set_xlabel(f'PC1 ({ve[0]:.1%})')
        ax.set_ylabel(f'PC2 ({ve[1]:.1%})')
        ax.set_title(nome)
        ax.legend(markerscale=2)
        rec = recall_centroide(Z, y_tr)
        print(f'    {nome}: ' + ', '.join(f'{c} = {r:.1%}' for c, r in rec.items()))
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
 
 

#  Tratamento das variáveis categóricas
def exp_categoricas(n_tr, n_te, c_tr, c_te, alvo):
    """n_tr/n_te: numéricas padronizadas. c_tr/c_te: categóricas após one-hot."""
    print('\n[5] Tratamento das variáveis categóricas')
 
    # decisão adotada: PCA só no bloco numérico, one-hot concatenado depois
    media, W, ve = pca_ajustar(n_tr, None)
    d = escolher_d(ve, alvo)
    Z_tr = pca_projetar(n_tr, media, W[:, :d])
    Z_te = pca_projetar(n_te, media, W[:, :d])
    final_tr = np.hstack([Z_tr, c_tr])
    final_te = np.hstack([Z_te, c_te])
    print(f'    Decisão adotada: PCA só no bloco numérico ({n_tr.shape[1]} colunas -> d = {d}) '
          f'+ {c_tr.shape[1]} colunas one-hot concatenadas, sem PCA')
    print(f'    Representação final: treino {final_tr.shape}, teste {final_te.shape}')
 
    # alternativa descartada: projetar tudo (one-hot incluso, tudo padronizado)
    todos = np.hstack([n_tr, c_tr])
    X = StandardScaler().fit_transform(todos)
    _, W_t, ve_t = pca_ajustar(X, None)
    d_t = escolher_d(ve_t, alvo)
    n_num = n_tr.shape[1]
    print(f'    Alternativa (projetar tudo, one-hot padronizado): {X.shape[1]} colunas, '
          f'd({alvo:.0%}) = {d_t} (contra d = {d} só no bloco numérico)')
    for j in range(3):
        peso_cat = float((W_t[n_num:, j] ** 2).sum())
        print(f'      PC{j + 1}: {ve_t[j]:.1%} da variância; '
              f'{peso_cat:.0%} do peso da componente vem das colunas one-hot')
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dados', default='data.csv')
    ap.add_argument('--cenario', default='C', choices=['A', 'B', 'C'])
    ap.add_argument('--var', type=float, default=0.90,
                    help='variância acumulada alvo para escolher d (0.90 ou 0.95)')
    ap.add_argument('--mostrar', action='store_true', help='abrir os gráficos na tela')
    args = ap.parse_args()
 
    if not args.mostrar:
        plt.switch_backend('Agg')
    os.makedirs(PASTA_SAIDA, exist_ok=True)
 
    x, y = carregar_dados(args.dados, args.cenario)
    print(f'Cenário {args.cenario}: {x.shape[0]} alunos, {x.shape[1]} atributos')
    verificar_escala_das_notas(x)
 
    x_tr, x_te, y_tr, y_te = separar_treino_teste(x, y, seed=SEED)
    num, cat = dividir_blocos(list(x.columns))
    print(f'Bloco numérico (vai ao PCA): {len(num)} colunas | '
          f'bloco categórico (one-hot): {len(cat)} colunas')
    if len(num) < 2:
        raise SystemExit('Bloco numérico com menos de 2 colunas: PCA não se aplica.')
 
    c_tr, c_te, nomes_cat = codificar_categoricas(x_tr[cat], x_te[cat])
    print(f'Colunas após one-hot: {len(nomes_cat)}')
 
    dados_num = {nome: preparar_numericas(x_tr[num], x_te[num], padr)
                 for nome, padr in REGIMES.items()}
    n_tr_std, n_te_std = dados_num['Com padronização']
 
    exp_validacao(n_tr_std)
    exp_padronizacao(x_tr[num], dados_num, num, PASTA_SAIDA)
 
    ves = {nome: pca_ajustar(n_tr, None)[2] for nome, (n_tr, _) in dados_num.items()}
    exp_variancia(ves, args.var, os.path.join(PASTA_SAIDA, 'e2_variancia_explicada.png'))
 
    exp_projecao_2d(dados_num, y_tr, os.path.join(PASTA_SAIDA, 'e2_projecao_2d.png'))
    exp_categoricas(n_tr_std, n_te_std, c_tr, c_te, args.var)
 
    print(f'\nArquivos salvos em "{PASTA_SAIDA}/".')
    if args.mostrar:
        plt.show()
 
 
if __name__ == '__main__':
    main()
 

