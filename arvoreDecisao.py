import numpy as np

# calculo da entropia de Shannon: H = -soma(p * log2(p))
def entropia(y):
    if len(y) == 0:
        return 0.0
    _, contagens = np.unique(y, return_counts=True)
    probabilidades = contagens / len(y)
    return float(
        -np.sum(probabilidades * np.log2(probabilidades) + 0.0)
    )

# calculo do indice de Gini: G = 1 - soma(p^2)
def indice_gini(y):
    if len(y) == 0:
        return 0.0
    _, contagens = np.unique(y, return_counts=True)
    probabilidades = contagens / len(y)
    return float(1 - np.sum(probabilidades ** 2))

# calculo da impureza de um no, que pode ser entropia ou gini
def impureza(y, criterio):
    if criterio == "entropia":
        return entropia(y)
    if criterio == "gini":
        return indice_gini(y)
    raise ValueError("Critério deve ser 'entropia' ou 'gini'.")

# ganho de informacao ao dividir um no em dois filhos, esquerda e direita
def ganho_informacao(coluna, y, limiar, criterio="entropia"):
    coluna = np.asarray(coluna)
    y = np.asarray(y)

    mascara = coluna <= limiar
    esquerda = y[mascara]
    direita = y[~mascara]

    if len(esquerda) == 0 or len(direita) == 0:
        return 0.0

    impureza_pai = impureza(y, criterio)

    impureza_filhos = (
        len(esquerda) / len(y) * impureza(esquerda, criterio)
        + len(direita) / len(y) * impureza(direita, criterio)
    )

    return float(impureza_pai - impureza_filhos)

# estrutura do nó
class No:
    def __init__(self, classe, impureza, profundidade=0, n=0, acertos=0):
        self.classe = classe
        self.impureza = float(impureza)
        self.profundidade = profundidade

        # usados pelas regras SE-ENTAO: n = exemplos que caem no no
        # acertos = quantos deles sao da classe majoritaria
        self.n = int(n)
        self.acertos = int(acertos)

        self.atributo = None
        self.limiar = None
        self.esq = None
        self.dir = None
        self.ganho = 0.0

    def folha(self):
        return self.atributo is None

    def virar_folha(self):
        self.atributo = None
        self.limiar = None
        self.esq = None
        self.dir = None
        self.ganho = 0.0

# retorna o atributo, o limiar e a redução de impureza da melhor divisão
def encontrar_melhor_divisao(X, y, criterio, min_folha):
    melhor_score = float("inf")
    melhor_atributo = None
    melhor_limiar = None

    n_amostras, n_atributos = X.shape
    impureza_pai = impureza(y, criterio)

    for j in range(n_atributos):
        valores = np.unique(X[:, j])

        if len(valores) < 2:
            continue

        limiares = (valores[:-1] + valores[1:]) / 2

        for limiar in limiares:
            mascara = X[:, j] <= limiar
            n_esq = mascara.sum()
            n_dir = n_amostras - n_esq

            # impede folhas com poucos exemplos
            if n_esq < min_folha or n_dir < min_folha:
                continue

            y_esq = y[mascara]
            y_dir = y[~mascara]

            score = (
                (len(y_esq) / n_amostras)
                * impureza(y_esq, criterio)
                + (len(y_dir) / n_amostras)
                * impureza(y_dir, criterio)
            )

            if score < melhor_score:
                melhor_score = score
                melhor_atributo = j
                melhor_limiar = float(limiar)

    if melhor_atributo is None:
        return None, None, 0.0

    reducao = impureza_pai - melhor_score

    return melhor_atributo, melhor_limiar, reducao


def classe_majoritaria(y):
    classes, contagens = np.unique(y, return_counts=True)
    return classes[np.argmax(contagens)]


def induzir_arvore(X, y, criterio="gini", prof_max=5, min_folha=5, profundidade=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
 
    majoritaria = classe_majoritaria(y)
 
    no = No(
        classe=majoritaria,
        impureza=impureza(y, criterio),
        profundidade=profundidade,
        n=len(y),
        acertos=int(np.sum(y == majoritaria))
    )
 
    # condição 1: nó puro
    if len(np.unique(y)) == 1:
        return no
 
    # condição 2: profundidade máxima (prof_max=None significa sem limite)
    if prof_max is not None and profundidade >= prof_max:
        return no
 
    # condição 3: não há dados suficientes para dividir
    if len(y) < 2 * min_folha:
        return no
 
    atributo, limiar, reducao = encontrar_melhor_divisao(
        X, y, criterio, min_folha
    )
 
    # condição 4: não existe divisão útil
    if atributo is None or reducao <= 1e-12:
        return no
 
    mascara = X[:, atributo] <= limiar
 
    no.atributo = atributo
    no.limiar = limiar
 
    no.esq = induzir_arvore(
        X[mascara], y[mascara],
        criterio, prof_max, min_folha, profundidade + 1
    )
 
    no.dir = induzir_arvore(
        X[~mascara], y[~mascara],
        criterio, prof_max, min_folha, profundidade + 1
    )
 
    return no


def prever_um(no, amostra, prof_corte=None):
    if no.folha():
        return no.classe
 
    if prof_corte is not None and no.profundidade >= prof_corte:
        return no.classe
 
    if amostra[no.atributo] <= no.limiar:
        return prever_um(no.esq, amostra, prof_corte)
 
    return prever_um(no.dir, amostra, prof_corte)


def prever_arvore(raiz, X, prof_corte=None):
    X = np.asarray(X, dtype=float)
    return np.array([prever_um(raiz, x, prof_corte) for x in X])

def contar_erros(no, X, y):
    if len(X) == 0:
        return 0
 
    if no.folha():
        return int(np.sum(y != no.classe))
 
    mascara = X[:, no.atributo] <= no.limiar
 
    return (
        contar_erros(no.esq, X[mascara], y[mascara])
        + contar_erros(no.dir, X[~mascara], y[~mascara])
    )

def podar_no(no, X, y):
    if no.folha():
        return
 
    mascara = X[:, no.atributo] <= no.limiar
 
    podar_no(no.esq, X[mascara], y[mascara])
    podar_no(no.dir, X[~mascara], y[~mascara])
 
    if len(X) == 0:
        no.virar_folha()
        return
 
    erros_como_folha = int(np.sum(y != no.classe))
 
    if contar_erros(no, X, y) >= erros_como_folha:
        no.virar_folha()
 
 
def podar_arvore(raiz, X_val, y_val):
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val)
    podar_no(raiz, X_val, y_val)
    return raiz

def extrair_regras(raiz, nomes_atributos=None):
    regras = []
    n_raiz = raiz.n
 
    def caminhar(no, condicoes):
        if no.folha():
            regras.append({
                "condicoes": list(condicoes),
                "classe": no.classe,
                "n": no.n,
                "cobertura": no.n / n_raiz,
                "confianca": no.acertos / no.n if no.n > 0 else 0.0,
            })
            return
 
        if nomes_atributos is None:
            nome = f"x[{no.atributo}]"
        else:
            nome = str(nomes_atributos[no.atributo])
 
        caminhar(no.esq, condicoes + [f"{nome} <= {no.limiar:.4g}"])
        caminhar(no.dir, condicoes + [f"{nome} > {no.limiar:.4g}"])
 
    caminhar(raiz, [])
    return regras
 
 
def formatar_regras(regras, top=None):
    regras = sorted(regras, key=lambda r: (-r["cobertura"], -r["confianca"]))
 
    if top is not None:
        regras = regras[:top]
 
    linhas = []
    for i, r in enumerate(regras, start=1):
        condicoes = " E ".join(r["condicoes"]) if r["condicoes"] else "(sempre)"
        linhas.append(
            f"R{i}: SE {condicoes}\n"
            f"     ENTAO {r['classe']} "
            f"[cobertura={r['cobertura']:.1%}, "
            f"confianca={r['confianca']:.1%}, n={r['n']}]"
        )
    return "\n".join(linhas)

# inspeção da árvore: quantidade de nós, profundidade máxima, quantidade de folhas
def profundidade_arvore(no):
    if no.folha():
        return 0
    return 1 + max(profundidade_arvore(no.esq), profundidade_arvore(no.dir))
 
 
def contar_folhas(no):
    if no.folha():
        return 1
    return contar_folhas(no.esq) + contar_folhas(no.dir)
 
 
def imprimir_arvore(no, nomes_atributos=None, nomes_classes=None, recuo=""):
    if no.folha():
        if nomes_classes is None:
            classe = no.classe
        else:
            classe = nomes_classes[no.classe]
        return f"{recuo}--> {classe} (n={no.n}, impureza={no.impureza:.3f})\n"
 
    if nomes_atributos is None:
        nome = f"x[{no.atributo}]"
    else:
        nome = str(nomes_atributos[no.atributo])
 
    texto = f"{recuo}[{nome} <= {no.limiar:.4g}] n={no.n}\n"
    texto += imprimir_arvore(no.esq, nomes_atributos, nomes_classes, recuo + "   ")
    texto += imprimir_arvore(no.dir, nomes_atributos, nomes_classes, recuo + "   ")
    return texto