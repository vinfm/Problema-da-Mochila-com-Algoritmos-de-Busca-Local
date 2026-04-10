import math
import os
import random
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MAX_GERACOES = 600
NUM_ITENS = 200
N = NUM_ITENS*2
R = N
pCROSS = 0.8
pMUT = 0.05
ValorFITNESS_OK = float("inf")  # Pode ser ajustado para um valor específico se desejado
MAX_ITENS_COMPARACAO_EXATA = 500
MAX_PESO_ITEM = 100
MAX_VALOR_ITEM = 100
MIN_PESO_ITEM = 30
MIN_VALOR_ITEM = 5
CAPACIDADE = (MAX_PESO_ITEM - MIN_PESO_ITEM) * NUM_ITENS // 2  # Capacidade média baseada nos itens gerados
MAX_CUSTO_DP_EXATA = MAX_ITENS_COMPARACAO_EXATA * (CAPACIDADE + 1)

class Estado:
    def __init__(self, solucao):
        self.solucao = solucao
        self.valor_fitness = None
        self.peso_total = None

def valor_total(sol, itens):
    """Soma os valores dos itens selecionados"""
    return sum(sol[i] * itens[i]["valor"] for i in range(len(itens)))

def peso_total(sol, itens):
    """Soma os pesos dos itens selecionados"""
    return sum(sol[i] * itens[i]["peso"] for i in range(len(itens)))

def solucao_valida(estado_ou_solucao, itens, capacidade):
    """Verifica se o peso total não excede a capacidade."""
    if isinstance(estado_ou_solucao, Estado):
        if estado_ou_solucao.peso_total is None:
            estado_ou_solucao.peso_total = peso_total(estado_ou_solucao.solucao, itens)
        return estado_ou_solucao.peso_total <= capacidade
    return peso_total(estado_ou_solucao, itens) <= capacidade

def conserta_solucao(estado, itens, capacidade):
    """Conserta um estado inválido removendo itens aleatórios até ficar válido."""
    if estado.peso_total is None:
        estado.peso_total = peso_total(estado.solucao, itens)

    while estado.peso_total > capacidade:
        indices_ativos = [idx for idx, bit in enumerate(estado.solucao) if bit == 1]
        if not indices_ativos:
            break
        i = random.choice(indices_ativos)
        estado.solucao[i] = 0
        estado.peso_total -= itens[i]["peso"]

def calcula_fitness(estado, itens, capacidade):
    """Calcula o fitness do estado, penalizando soluções inválidas."""
    if solucao_valida(estado, itens, capacidade):
        if estado.valor_fitness is None:
            estado.valor_fitness = valor_total(estado.solucao, itens)
        return estado.valor_fitness
    return 0

def selecao_roleta(populacao, total_fitness):
    """Seleciona um indivíduo da população usando seleção por roleta"""
    
    if total_fitness <= 0:
        return random.choice(populacao)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind in populacao:
        current += ind.valor_fitness
        if current >= pick:
            return ind

def selecao(populacao):
    """Seleciona um indivíduo da população usando torneio"""
    nova_populacao = []
    total_fitness = sum(ind.valor_fitness for ind in populacao)
    for _ in range(R):
        nova_populacao.append(selecao_roleta(populacao, total_fitness))
    return nova_populacao

def crossover(pai1, pai2, pCROSS):
    """Realiza o crossover entre dois pais para gerar um filho"""
    if random.random() < pCROSS:
        ponto_corte = random.randint(1, len(pai1) - 2)
        filho = pai1[:ponto_corte] + pai2[ponto_corte:]
    else:
        filho = pai1[:]
    return filho

def crossovers(populacao, pCROSS):
    """Realiza o crossover em toda a população para gerar uma nova população"""
    nova_populacao = []
    i = 0
    while i < R - 1:
        filho1_solucao = crossover(populacao[i].solucao, populacao[i + 1].solucao, pCROSS)
        filho2_solucao = crossover(populacao[i + 1].solucao, populacao[i].solucao, pCROSS)
        nova_populacao.append(Estado(filho1_solucao))
        nova_populacao.append(Estado(filho2_solucao))
        i += 1
    return nova_populacao

def mutacao(solucao, pMUT):
    """Realiza a mutação em uma solução com probabilidade pMUT"""
    for i in range(len(solucao)):
        if random.random() < pMUT:
            solucao[i] = 1 - solucao[i]  # Inverte o bit
    return solucao

def selecao_melhores(populacao, R):
    """Seleciona os R melhores indivíduos da população"""
    return sorted(populacao, key=lambda ind: ind.valor_fitness, reverse=True)[:R]

def fitness_estagnou(populacao):
    """Verifica se o fitness estagnou nos últimos estagios gerações"""
    melhores = sorted(populacao, key=lambda ind: ind.valor_fitness, reverse=True)[0:int(N * 0.9)]
    return all(ind.valor_fitness == melhores[0].valor_fitness for ind in melhores)

def distancia_media_ao_melhor(populacao):
    """Mede diversidade como distância de Hamming média até o melhor indivíduo."""
    if not populacao:
        return 0.0

    melhor = max(populacao, key=lambda ind: ind.valor_fitness)
    n_bits = len(melhor.solucao)
    if n_bits == 0:
        return 0.0

    soma_distancias_norm = 0.0
    for ind in populacao:
        distancia = sum(1 for a, b in zip(ind.solucao, melhor.solucao) if a != b)
        soma_distancias_norm += distancia / n_bits

    return soma_distancias_norm / len(populacao)

def AG(C, pCROSS, pMUT, N, R, itens, capacidade):
    # N e R passam a refletir automaticamente o tamanho atual de C, se necessário.
    N = len(C) if N is None else N
    R = N if R is None else R

    hist_melhor_val = []
    hist_fitness_medio = []
    hist_pior_val = []
    hist_distancia_media = []
    hist_geracao = []

    for c in C:
        c.peso_total = peso_total(c.solucao, itens)
        if not solucao_valida(c, itens, capacidade):
            conserta_solucao(c, itens, capacidade)
        c.valor_fitness = calcula_fitness(c, itens, capacidade)

    geracao = 0
    while (1):
        nova_populacao = []
        nova_populacao = selecao(C)

        nova_populacao_cruzada = []
        nova_populacao_cruzada = crossovers(nova_populacao, pCROSS)
        for ind in nova_populacao_cruzada:
            ind.solucao = mutacao(ind.solucao, pMUT)

        for ind in nova_populacao_cruzada:
            ind.peso_total = peso_total(ind.solucao, itens)
            if not solucao_valida(ind, itens, capacidade):
                conserta_solucao(ind, itens, capacidade)
            ind.valor_fitness = calcula_fitness(ind, itens, capacidade)

        C = selecao_melhores(C + nova_populacao_cruzada, R)

        melhor_geracao = C[0].valor_fitness
        pior_geracao = C[-1].valor_fitness
        media_geracao = sum(ind.valor_fitness for ind in C) / len(C)

        hist_melhor_val.append(melhor_geracao)
        hist_fitness_medio.append(media_geracao)
        hist_pior_val.append(pior_geracao)
        hist_distancia_media.append(distancia_media_ao_melhor(C))
        hist_geracao.append(geracao)

        geracao += 1
        if (geracao >= MAX_GERACOES or C[0].valor_fitness >= ValorFITNESS_OK or fitness_estagnou(C)):
            break

    sorted_populacao = sorted(C, key=lambda ind: ind.valor_fitness, reverse=True)
    historico = {
        "geracao": hist_geracao,
        "melhor_val": hist_melhor_val,
        "fitness_medio": hist_fitness_medio,
        "pior_val": hist_pior_val,
        "distancia_media": hist_distancia_media,
    }
    return sorted_populacao[0], historico


##### Funções para comparação com o ótimo global exato via programação dinâmica
def maximo_global_mochila(itens, capacidade):
    """Resolve a mochila 0/1 exatamente via programação dinâmica e reconstrói a solução ótima."""
    n = len(itens)
    dp = [[0] * (capacidade + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        peso_i = itens[i - 1]["peso"]
        valor_i = itens[i - 1]["valor"]
        for cap in range(capacidade + 1):
            melhor_sem_item = dp[i - 1][cap]
            melhor_com_item = -1
            if peso_i <= cap:
                melhor_com_item = dp[i - 1][cap - peso_i] + valor_i
            dp[i][cap] = max(melhor_sem_item, melhor_com_item)

    solucao_otima = [0] * n
    cap = capacidade
    for i in range(n, 0, -1):
        if dp[i][cap] != dp[i - 1][cap]:
            solucao_otima[i - 1] = 1
            cap -= itens[i - 1]["peso"]

    valor_otimo = dp[n][capacidade]
    peso_otimo = peso_total(solucao_otima, itens)
    return solucao_otima, valor_otimo, peso_otimo

def comparar_ag_com_otimo(melhor_solucao_ag, itens, capacidade):
    """Compara o resultado do AG com o ótimo global exato da mochila."""
    melhor_valor_ag = valor_total(melhor_solucao_ag, itens)
    melhor_peso_ag = peso_total(melhor_solucao_ag, itens)

    custo_dp = len(itens) * (capacidade + 1)

    if custo_dp > MAX_CUSTO_DP_EXATA:
        return {
            "valor_ag": melhor_valor_ag,
            "peso_ag": melhor_peso_ag,
            "comparacao_exata": False,
            "motivo": (
                f"Comparacao exata pulada: custo DP={custo_dp} "
                f"> limite {MAX_CUSTO_DP_EXATA} (MAX_ITENS*CAPACIDADE)"
            ),
        }

    solucao_otima, valor_otimo, peso_otimo = maximo_global_mochila(itens, capacidade)
    gap_abs = valor_otimo - melhor_valor_ag
    gap_pct = (gap_abs / valor_otimo * 100) if valor_otimo > 0 else 0.0

    return {
        "valor_ag": melhor_valor_ag,
        "peso_ag": melhor_peso_ag,
        "comparacao_exata": True,
        "solucao_otima": solucao_otima,
        "valor_otimo": valor_otimo,
        "peso_otimo": peso_otimo,
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
    }

# Parte de Plotagem
def plotar(
    historico,
    itens,
    melhor,
    capacidade,
    comparacao=None,
    tempo_ag=None,
    arquivo_saida="resultado_ag_knapsack.png",
    exibir=True,
):
    """Gera gráficos explicativos do comportamento do AG."""
    geracoes = historico["geracao"]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Problema da Mochila Binária - Algoritmo Genético",
                 fontsize=14, fontweight="bold", y=0.98)
    parametros_figura = (
        f"População = {N}   |   Máx. gerações = {MAX_GERACOES}   |   "
        f"Itens = {NUM_ITENS}   |   Capacidade = {capacidade}   |   "
        f"pCROSS = {pCROSS:.2f}   |   pMUT = {pMUT:.2f}"
    )
    fig.text(
        0.5,
        0.945,
        parametros_figura,
        ha="center",
        fontsize=9,
        color="#444",
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    AZUL = "#185FA5"
    VERDE = "#1D9E75"
    AMBAR = "#BA7517"
    ROXO = "#7F77DD"
    CINZA = "#888780"

    # Evolução dos valores de fitness
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(geracoes, historico["melhor_val"], color=VERDE, lw=2.0, label="Melhor fitness")
    ax1.plot(geracoes, historico["fitness_medio"], color=AZUL, lw=1.5, alpha=0.8, label="Fitness médio")
    ax1.plot(geracoes, historico["pior_val"], color=CINZA, lw=1.2, alpha=0.8, label="Pior fitness")
    ax1.set_title("Evolução do fitness por geração", fontsize=11)
    ax1.set_xlabel("Geração", fontsize=9)
    ax1.set_ylabel("Fitness", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=8)

    # Distância média ao melhor (diversidade por Hamming)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(geracoes, historico["distancia_media"], color=ROXO, lw=2.0)
    ax2.fill_between(geracoes, historico["distancia_media"], alpha=0.15, color=ROXO)
    ax2.set_title("Distância média ao melhor (Hamming)", fontsize=11)
    ax2.set_xlabel("Geração", fontsize=9)
    ax2.set_ylabel("Distância normalizada (0 a 1)", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)

    # Melhor fitness acumulado
    ax3 = fig.add_subplot(gs[1, 0])
    melhor_acumulado = []
    melhor_ate_agora = float("-inf")
    for valor in historico["melhor_val"]:
        melhor_ate_agora = max(melhor_ate_agora, valor)
        melhor_acumulado.append(melhor_ate_agora)
    ax3.plot(geracoes, melhor_acumulado, color=AMBAR, lw=2.0)
    ax3.fill_between(geracoes, melhor_acumulado, alpha=0.15, color=AMBAR)
    ax3.set_title("Melhor fitness acumulado", fontsize=11)
    ax3.set_xlabel("Geração", fontsize=9)
    ax3.set_ylabel("Melhor fitness", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=8)

    # Itens selecionados na melhor solução
    ax4 = fig.add_subplot(gs[1, 1])
    xs_out, ys_out = [], []
    xs_in, ys_in = [], []
    for i, it in enumerate(itens):
        if melhor[i]:
            xs_in.append(it["peso"])
            ys_in.append(it["valor"])
        else:
            xs_out.append(it["peso"])
            ys_out.append(it["valor"])

    ax4.scatter(xs_out, ys_out, color=CINZA, alpha=0.5, s=50, label="Não selecionado", zorder=2)
    ax4.scatter(xs_in, ys_in, color=VERDE, alpha=0.9, s=90, label="Selecionado", zorder=3, edgecolors="white", lw=0.8)
    peso_usado = peso_total(melhor, itens)
    ax4.axvline(capacidade, color=AMBAR, lw=1.5, ls="--", label=f"Capacidade={capacidade}")
    ax4.set_title("Itens: peso × valor (solução final)", fontsize=11)
    ax4.set_xlabel("Peso", fontsize=9)
    ax4.set_ylabel("Valor", fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=8)
    
    # Encontra a geração onde o melhor valor se estabiliza (deixa de variar)
    melhor_val_historico = historico['melhor_val']
    geracao_convergencia = None
    
    # Percorre do final para o início para encontrar onde começou a ser constante
    valor_estavel = melhor_val_historico[-1]
    for i in range(len(melhor_val_historico) - 1, -1, -1):
        if melhor_val_historico[i] != valor_estavel:
            geracao_convergencia = historico['geracao'][i + 1]
            break
    
    # Se nunca variou, começou constante desde o início
    if geracao_convergencia is None:
        geracao_convergencia = historico['geracao'][0]

    convergencia_texto = f"   |   Convergência na geração {geracao_convergencia}"
    
    # Rodapé
    val_final = valor_total(melhor, itens)
    tempo_decorrido = f"   |   Tempo AG = {tempo_ag:.2f}s" if tempo_ag is not None else ""
    texto_rodape = (
        f"Melhor valor = {val_final}   |   Peso usado = {peso_usado}/{capacidade}   |   "
        f"Itens selecionados = {sum(melhor)}/{len(itens)}{convergencia_texto}{tempo_decorrido}"
    )

    if comparacao is not None:
        if comparacao.get("comparacao_exata"):
            texto_rodape += (
                f"\nOtimo global = {comparacao['valor_otimo']} (peso {comparacao['peso_otimo']})"
                f"   |   Gap AG = {comparacao['gap_abs']} ({comparacao['gap_pct']:.2f}%)"
            )
        else:
            texto_rodape += f"\n{comparacao.get('motivo', 'Comparacao exata nao realizada')}"

    fig.text(0.5, 0.01, texto_rodape, ha="center", fontsize=10, color="#333")

    diretorio_saida = os.path.dirname(arquivo_saida)
    if diretorio_saida:
        os.makedirs(diretorio_saida, exist_ok=True)

    plt.savefig(arquivo_saida, dpi=150, bbox_inches="tight")
    if exibir:
        plt.show()
    else:
        plt.close(fig)

# Geração da população inicial e execução do AG

def gerar_solucao_aleatoria():
    """Gera uma solução aleatória (vetor binário)"""
    return [random.randint(0, 1) for _ in range(NUM_ITENS)]

def gerar_populacao_inicial(N):
    """Gera uma população inicial de soluções aleatórias"""
    return [Estado(gerar_solucao_aleatoria()) for _ in range(N)]

if __name__ == "__main__":
    # Cada item tem nome, peso >= 0 e valor >= 0
    random.seed()
    itens = [
        {"nome": f"Item_{i:02d}", "peso": random.randint(MIN_PESO_ITEM, MAX_PESO_ITEM), "valor": random.randint(MIN_VALOR_ITEM, MAX_VALOR_ITEM)}
        for i in range(NUM_ITENS)
    ]

    C = gerar_populacao_inicial(N)
    tempo_ag_inicio = time.time()
    melhor, historico = AG(C, pCROSS, pMUT, N, R, itens, CAPACIDADE)
    tempo_ag = time.time() - tempo_ag_inicio
    comparacao = comparar_ag_com_otimo(melhor.solucao, itens, CAPACIDADE)

    plotar(historico, itens, melhor.solucao, CAPACIDADE, comparacao, tempo_ag)