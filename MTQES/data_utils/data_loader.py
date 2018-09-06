import os

fen = open('../data/dev/dev.source', encoding='utf8')
fes = open('../data/dev/dev.target', encoding='utf8')


res = []
for en_line, es_line in zip(fen, fes):
    res.append(''.join([en_line.strip('\n'), '\t', es_line]))

fen.close()
fes.close()

with open('../data/dev/dev.tsv', 'w', encoding='utf8') as fout:
    fout.writelines(res)
