import math
import os
import random
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import statistics

def valor_total(sol, itens):
    """Soma os valores dos itens selecionados"""
    return sum(sol[i] * itens[i]["valor"] for i in range(len(itens)))


def peso_total(sol, itens):
    """Soma os pesos dos itens selecionados"""
    return sum(sol[i] * itens[i]["peso"] for i in range(len(itens)))


def solucao_valida(sol, itens, capacidade):
    """Verifica se o peso total não excede a capacidade"""
    return peso_total(sol, itens) <= capacidade

def gerar_vizinho(sol, itens, capacidade):
    """Gera um vizinho invertendo um bit aleatório"""
    n = len(sol)
    vizinho = sol[:]
    i = random.randint(0, n - 1)
    vizinho[i] = 1 - vizinho[i]

    if solucao_valida(vizinho, itens, capacidade):
        return vizinho

    # Se adicionou um item e passar da capacidade reverte 
    vizinho[i] = sol[i]
    return vizinho


def probabilidade_aceitar(delta, temperatura):
    """Retorna a probabilidade de aceitar uma solução pior"""
    if temperatura < 1e-10:
        return 0.0
    return math.exp(delta / temperatura)

def simulated_annealing(
    itens,
    capacidade,
    T0,          # Temperatura inicial
    Tn,             # Temperatura final 
    n_fases,        # Controla duração do resfriamento
    iter_por_temp,   # Iterações por nível de temperatura
    seed = None,
    solucao_inicial=None,
):
    """
    Executa o Simulated Annealing para o Problema da Mochila Binária.

    Resfriamento quadrático:
        T_t = Tn + (T0 - Tn) * ((n - t) / n)^2
    onde t é a fase atual (0..n-1) e n = n_fases.

    Parâmetros
    ----------
    itens         : lista de dicts com chaves 'nome', 'peso', 'valor'
    capacidade    : capacidade máxima da mochila
    T0            : temperatura inicial (t=0)
    Tn            : temperatura final desejada (t=n), deve ser > 0
    n_fases       : número total de fases; substitui alpha — quanto maior,
                    mais lento o resfriamento e maior a exploração
    iter_por_temp : quantas vizinhanças são exploradas por temperatura
    seed          : semente aleatória para reprodutibilidade

    Retorna
    -------
    melhor_sol    : melhor solução encontrada
    historico     : dict com séries temporais para plotagem
    """
    if seed is not None:
        random.seed(seed)

    n = len(itens)

    # Mochila vazia
    atual = [0] * n
    val_atual = 0

    melhor = atual[:]
    melhor_val = 0

    T = T0

    # Históricos para os gráficos
    hist_val_atual  = []
    hist_melhor_val = []
    hist_temperatura = []
    hist_taxa_aceite = []
    hist_iters       = []
    iter_global = 0

    fase = 0

    # Loop de resfriamento
    while fase < n_fases:
        aceitos = 0
        pioras_aceitas = 0
        pioras_tentadas = 0

        # Iterações por temperatura
        for _ in range(iter_por_temp):
            vizinho = gerar_vizinho(atual, itens, capacidade)
            val_vizinho = valor_total(vizinho, itens)
            delta = val_vizinho - val_atual

            if delta >= 0:
                # Melhoria: aceita sempre
                atual = vizinho
                val_atual = val_vizinho
                aceitos += 1
            else:
                # Piora: aceita com probabilidade e^(delta/T)
                pioras_tentadas += 1
                prob = probabilidade_aceitar(delta, T)
                if random.random() < prob:
                    atual = vizinho
                    val_atual = val_vizinho
                    aceitos += 1
                    pioras_aceitas += 1

            # Atualiza o melhor global
            if val_atual > melhor_val:
                melhor = atual[:]
                melhor_val = val_atual


            iter_global += 1

        # Registra histórico da fase
        taxa = (pioras_aceitas / pioras_tentadas * 100) if pioras_tentadas > 0 else 0.0
        hist_val_atual.append(val_atual)
        hist_melhor_val.append(melhor_val)
        hist_temperatura.append(T)
        hist_taxa_aceite.append(taxa)
        hist_iters.append(iter_global)

        # Resfriamento quadrático
        fase += 1
        T = Tn + (T0 - Tn) * ((n_fases - fase) / n_fases) ** 2

    
    historico = {
        "val_atual":   hist_val_atual,
        "melhor_val":  hist_melhor_val,
        "temperatura": hist_temperatura,
        "taxa_aceite": hist_taxa_aceite,
        "iters":       hist_iters,
    }
    return melhor, historico

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

def comparar_sa_com_otimo(melhor_solucao_sa, itens, capacidade):
    """Compara o resultado da SA com o ótimo global exato da mochila."""
    melhor_valor_sa = valor_total(melhor_solucao_sa, itens)
    melhor_peso_sa = peso_total(melhor_solucao_sa, itens)

    custo_dp = len(itens) * (capacidade + 1)
    MAX_CUSTO_DP_EXATA = 300 * (capacidade + 1)

    if custo_dp > MAX_CUSTO_DP_EXATA:
        return {
            "valor_sa": melhor_valor_sa,
            "peso_sa": melhor_peso_sa,
            "comparacao_exata": False,
            "motivo": (
                f"Comparacao exata pulada: custo DP={custo_dp} "
                f"> limite {MAX_CUSTO_DP_EXATA} (MAX_ITENS*CAPACIDADE)"
            ),
        }

    solucao_otima, valor_otimo, peso_otimo = maximo_global_mochila(itens, capacidade)
    gap_abs = valor_otimo - melhor_valor_sa
    gap_pct = (gap_abs / valor_otimo * 100) if valor_otimo > 0 else 0.0

    return {
        "valor_sa": melhor_valor_sa,
        "peso_sa": melhor_peso_sa,
        "comparacao_exata": True,
        "solucao_otima": solucao_otima,
        "valor_otimo": valor_otimo,
        "peso_otimo": peso_otimo,
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
    }

def plotar(historico, itens, melhor, capacidade, tempo_inicio=None, comparacao=None,arquivo_saida="resultado_sa_knapsack.png",
    exibir=True,):
    """Gera 4 gráficos explicativos do comportamento da TS."""
    fases = list(range(len(historico["val_atual"])))

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Problema da Mochila Binária - Tempera Simulada",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    AZUL   = "#185FA5"
    VERDE  = "#1D9E75"
    AMBAR  = "#BA7517"
    ROXO   = "#7F77DD"
    CINZA  = "#888780"

    # Evolução do valor 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fases, historico["val_atual"],  color=AZUL,  lw=1.2, alpha=0.7, label="Valor atual")
    ax1.plot(fases, historico["melhor_val"], color=VERDE, lw=2.0, label="Melhor valor")
    ax1.set_title("Evolução do valor por fase", fontsize=11)
    ax1.set_xlabel("Fase", fontsize=9)
    ax1.set_ylabel("Valor total", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=8)

    # Temperatura ao longo das fase
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fases, historico["temperatura"], color=AMBAR, lw=2.0)
    ax2.fill_between(fases, historico["temperatura"], alpha=0.15, color=AMBAR)
    ax2.set_title("Resfriamento da temperatura", fontsize=11)
    ax2.set_xlabel("Fase", fontsize=9)
    ax2.set_ylabel("Temperatura T", fontsize=9)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.tick_params(labelsize=8)

    # Taxa de aceitação de pioras
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(fases, historico["taxa_aceite"], color=ROXO, alpha=0.75, width=1.0)
    ax3.set_title("Taxa de aceitação de pioras por fase (%)", fontsize=11)
    ax3.set_xlabel("Fase", fontsize=9)
    ax3.set_ylabel("% pioras aceitas", fontsize=9)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.tick_params(labelsize=8)

    # Itens selecionados
    ax4 = fig.add_subplot(gs[1, 1])
    xs_out, ys_out = [], []
    xs_in,  ys_in  = [], []
    for i, it in enumerate(itens):
        if melhor[i]:
            xs_in.append(it["peso"])
            ys_in.append(it["valor"])
        else:
            xs_out.append(it["peso"])
            ys_out.append(it["valor"])

    ax4.scatter(xs_out, ys_out, color=CINZA,  alpha=0.5, s=50,  label="Não selecionado", zorder=2)
    ax4.scatter(xs_in,  ys_in,  color=VERDE,  alpha=0.9, s=90,  label="Selecionado",     zorder=3, edgecolors="white", lw=0.8)

    peso_usado = peso_total(melhor, itens)
    ax4.axvline(capacidade, color=AMBAR, lw=1.5, ls="--", label=f"Capacidade={capacidade}")
    ax4.set_title("Itens: peso × valor (solução final)", fontsize=11)
    ax4.set_xlabel("Peso", fontsize=9)
    ax4.set_ylabel("Valor", fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=8)

    # Rodapé
    aux = len(historico["val_atual"]) - 1
    val_atualfinal = historico['val_atual'][aux]
    val_final = valor_total(melhor, itens)
    
    # Encontra a iteração onde o melhor valor se estabiliza (deixa de variar)
    melhor_val_historico = historico['melhor_val']
    iteracao_convergencia = None
    
    # Percorre do final para o início para encontrar onde começou a ser constante
    valor_estavel = melhor_val_historico[-1]
    for i in range(len(melhor_val_historico) - 1, -1, -1):
        if melhor_val_historico[i] != valor_estavel:
            iteracao_convergencia = historico['iters'][i + 1]
            break
    
    # Se nunca variou, começou constante desde o início
    if iteracao_convergencia is None:
        iteracao_convergencia = historico['iters'][0]
    
    tempo_decorrido = f"   |   Tempo de execução = {time.time() - tempo_inicio:.2f}s" if tempo_inicio else ""
    convergencia_texto = f"   |   Convergência na iteração {iteracao_convergencia}" if iteracao_convergencia else ""
    
    texto_rodape = (
        f"Melhor valor = {val_final} | Valor Atual = {val_atualfinal} |  Peso usado = {peso_usado}/{capacidade}   |"
        f"Itens selecionados = {sum(melhor)}/{len(itens)}{convergencia_texto}{tempo_decorrido}"
    )
    
    if comparacao is not None:
        if comparacao.get("comparacao_exata"):
            texto_rodape += (
                f"\nOtimo global = {comparacao['valor_otimo']} (peso {comparacao['peso_otimo']})"
                f"   |   Gap SA = {comparacao['gap_abs']} ({comparacao['gap_pct']:.2f}%)"
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

if __name__ == "__main__":
    tempo_inicio = time.time()

    # Cada item tem nome, peso >= 0 e valor >= 0
    random.seed(0)
    itens = [
        {"nome": f"Item_{i:02d}", "peso": random.randint(3, 25), "valor": random.randint(5, 50)}
        for i in range(100)
    ]

    capacidade = 500

    # GUIA DE PARÂMETROS
    #
    #  Resfriamento quadrático:
    #      T_t = Tn + (T0 - Tn) * ((n - t) / n)²
    #
    #  T0  (temperatura inicial)
    #      Alta  (ex: 1000-5000) -> aceita muitas pioras no início → mais exploração
    #      Baixa (ex: 10-50)    -> já começa conservador → pode ficar preso cedo
    #
    #  Tn  (temperatura final)
    #      Valor baixo próximo de zero (ex: 0.01 a 1.0)
    #      Garante que ao final o algoritmo não aceita mais pioras
    #
    #  n_fases  (número de fases — substitui alpha)
    #      Controla a duração total do resfriamento.
    #      Mais fases -> curva mais suave → mais exploração
    #      Valores típicos: 100 a 500 para instâncias de 10-50 itens
    #
    #  iter_por_temp  (iterações por temperatura)
    #      Mais iterações -> melhor amostragem em cada T, mas mais lento
    #      Valores típicos: 50 a 200 para instâncias de 10-50 itens

    melhor_sol, historico = simulated_annealing(
        itens         = itens,
        capacidade    = capacidade,
        T0            = 5000,  # Define exploração inicial
        Tn            = 0.1,     # Temperatura de parada
        n_fases       = 500,     # Controla duração do resfriamento
        iter_por_temp = 200,     # Vizinhanças avaliadas por nível de T
        seed          = None     
    )
    comparacao = comparar_sa_com_otimo(melhor_sol, itens, capacidade)
    plotar(historico, itens, melhor_sol, capacidade, tempo_inicio, comparacao)