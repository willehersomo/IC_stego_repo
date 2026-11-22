CONJUNTOS = ['7b']
for conj in CONJUNTOS:
    ctx = contexto(conj)
    print(f'\n################ {conj}: sanidade ################')
    for bpp in CARGAS:
        san = f45.sanidade(ctx['stc'], ctx['capas'], bpp, ctx['C'], n=32, processos=PROCESSOS)
        for nome, est in san.items():
            assert est['delta_max'] <= 1 and est['falhas'] <= 2, f'{conj} {nome}: sanidade do STC falhou: PARE'
    print(f'\n################ {conj}: corpus com STC real ################')
    for bpp in CARGAS:
        for m in ctx['stc']:
            f45.gerar_corpus_stc(m, ctx['capas'], ctx['idx_te'], bpp, ctx['C'], processos=PROCESSOS)
    copiar_simulados(conj, len(ctx['capas']))
    print(f'\n################ {conj}: medida ################')
    f40.rodar(ctx['sim'] + ctx['stc'], cargas=CARGAS, detectores=DETECTORES, sementes=SEMENTES, cfg=ctx['cfg'])
    resumir(conj)
