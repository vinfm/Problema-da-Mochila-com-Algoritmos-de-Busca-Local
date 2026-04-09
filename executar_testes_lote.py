import argparse
import os
import random
import time
from datetime import datetime

from AG import (
    AG,
    CAPACIDADE,
    MAX_PESO_ITEM,
    MAX_VALOR_ITEM,
    MIN_PESO_ITEM,
    MIN_VALOR_ITEM,
    N,
    NUM_ITENS,
    R,
    Estado,
    comparar_ag_com_otimo,
    pCROSS,
    pMUT,
    plotar as plotar_ag,
    valor_total,
)
from TemperaSimulada import plotar as plotar_ts
from TemperaSimulada import simulated_annealing


def gerar_itens_aleatorios(quantidade, seed=None):
    if seed is None:
        random.seed()
        rng = random
    else:
        rng = random.Random(seed)
    return [
        {
            "nome": f"Item_{i:02d}",
            "peso": rng.randint(MIN_PESO_ITEM, MAX_PESO_ITEM),
            "valor": rng.randint(MIN_VALOR_ITEM, MAX_VALOR_ITEM),
        }
        for i in range(quantidade)
    ]


def gerar_vetores_bits(num_testes, tamanho_vetor, seed):
    rng = random.Random(seed)
    return [[rng.randint(0, 1) for _ in range(tamanho_vetor)] for _ in range(num_testes)]


def gerar_populacao_com_vetor(vetor_base, tamanho_populacao, seed):
    rng = random.Random(seed)
    populacao = [Estado(vetor_base[:])]

    while len(populacao) < tamanho_populacao:
        individuo = [rng.randint(0, 1) for _ in range(len(vetor_base))]
        populacao.append(Estado(individuo))

    return populacao


def caminho_saida(pasta_saida, prefixo, indice_teste, formato):
    nome_arquivo = f"{prefixo}_teste_{indice_teste:02d}.{formato}"
    return os.path.join(pasta_saida, nome_arquivo)


def executar_testes(
    num_testes,
    seed,
    formato,
    pasta_ag,
    pasta_ts,
    mostrar_graficos,
):
    os.makedirs(pasta_ag, exist_ok=True)
    os.makedirs(pasta_ts, exist_ok=True)

    print("Iniciando bateria de testes...")
    print(f"- Numero de testes: {num_testes}")
    print(f"- Tamanho do vetor (NUM_ITENS): {NUM_ITENS}")
    print(f"- Capacidade: {CAPACIDADE}")

    itens = gerar_itens_aleatorios(NUM_ITENS, seed)
    seed_bits = None if seed is None else seed + 1
    vetores_teste = gerar_vetores_bits(num_testes, NUM_ITENS, seed_bits)
    if seed is None:
        seed = random.randint(0, 1000000)
    
    # É fundamental imprimir a seed para caso você precise repetir um teste específico
    print(f"Semente utilizada: {seed}")
    
    resultados = []

    for idx, vetor_bits in enumerate(vetores_teste, start=1):
        print(f"\n[TESTE {idx:02d}] Executando AG e TS...")

        # AG
        populacao_inicial = gerar_populacao_com_vetor(vetor_bits, N, seed + idx)
        inicio_ag = time.time()
        melhor_ag, historico_ag = AG(populacao_inicial, pCROSS, pMUT, N, R, itens, CAPACIDADE)
        tempo_ag = time.time() - inicio_ag
        comparacao_ag = comparar_ag_com_otimo(melhor_ag.solucao, itens, CAPACIDADE)

        arquivo_ag = caminho_saida(pasta_ag, "ag", idx, formato)
        plotar_ag(
            historico_ag,
            itens,
            melhor_ag.solucao,
            CAPACIDADE,
            comparacao=comparacao_ag,
            tempo_ag=tempo_ag,
            arquivo_saida=arquivo_ag,
            exibir=mostrar_graficos,
        )

        # TS
        inicio_ts = time.time()
        melhor_ts, historico_ts = simulated_annealing(
            itens=itens,
            capacidade=CAPACIDADE,
            T0=1000.0,
            Tn=0.1,
            n_fases=300,
            iter_por_temp=100,
            seed=seed + idx,
            solucao_inicial=vetor_bits,
        )
        tempo_ts = time.time() - inicio_ts

        arquivo_ts = caminho_saida(pasta_ts, "ts", idx, formato)
        plotar_ts(
            historico_ts,
            itens,
            melhor_ts,
            CAPACIDADE,
            tempo_inicio=inicio_ts,
            arquivo_saida=arquivo_ts,
            exibir=mostrar_graficos,
        )

        valor_ag = valor_total(melhor_ag.solucao, itens)
        valor_ts = valor_total(melhor_ts, itens)

        resultados.append(
            {
                "teste": idx,
                "valor_ag": valor_ag,
                "tempo_ag": tempo_ag,
                "valor_ts": valor_ts,
                "tempo_ts": tempo_ts,
                "arquivo_ag": arquivo_ag,
                "arquivo_ts": arquivo_ts,
            }
        )

        print(
            f"[TESTE {idx:02d}] AG valor={valor_ag} ({tempo_ag:.2f}s) | "
            f"TS valor={valor_ts} ({tempo_ts:.2f}s)"
        )
        print(f"[TESTE {idx:02d}] Graficos salvos em: {arquivo_ag} e {arquivo_ts}")

    print("\nResumo final:")
    for r in resultados:
        print(
            f"- Teste {r['teste']:02d}: "
            f"AG={r['valor_ag']} ({r['tempo_ag']:.2f}s), "
            f"TS={r['valor_ts']} ({r['tempo_ts']:.2f}s)"
        )


def parse_args():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pasta_padrao = os.path.join("Graficos", "TestesConjuntos", f"Testes_{timestamp}")
    
    parser = argparse.ArgumentParser(
        description="Executa AG e TS em lote com vetores de bits de tamanho NUM_ITENS."
    )
    parser.add_argument("--num-testes", type=int, default=5, help="Quantidade de testes.")
    parser.add_argument("--seed", type=int, default=None, help="Semente base aleatoria (se nao fornecida, usa aleatoria).")
    parser.add_argument(
        "--formato",
        choices=["png", "jpg"],
        default="png",
        help="Formato das imagens salvas.",
    )
    parser.add_argument(
        "--pasta-ag",
        default=os.path.join(pasta_padrao, "AG"),
        help="Pasta de saida dos graficos do AG.",
    )
    parser.add_argument(
        "--pasta-ts",
        default=os.path.join(pasta_padrao, "TS"),
        help="Pasta de saida dos graficos da Tempera Simulada.",
    )
    parser.add_argument(
        "--mostrar-graficos",
        action="store_true",
        help="Exibe os graficos na tela durante a execucao.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    executar_testes(
        num_testes=args.num_testes,
        seed=args.seed,
        formato=args.formato,
        pasta_ag=args.pasta_ag,
        pasta_ts=args.pasta_ts,
        mostrar_graficos=args.mostrar_graficos,
    )
