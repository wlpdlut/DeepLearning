import os

fname = 'v7'
fen = open('../data/train/{0}.source'.format(fname), encoding='utf8')
fes = open('../data/train/{0}.target'.format(fname), encoding='utf8')
#ftgt = open('../data/train/{0}.hter'.format(fname), encoding='utf8')


res = []
#tgt = []
for en_line, es_line in zip(fen, fes):
    if len(en_line.strip('\n').split()) in range(5,50) and \
       len(es_line.strip('\n').split()) in range(5,50):
            res.append(''.join([en_line.strip('\n'), '\t', es_line]))
            #tgt.append(tgt_line)
            if len(res) > 30000:
                break

fen.close()
fes.close()
#ftgt.close()

with open('../data/train/{0}.tsv'.format(fname), 'w', encoding='utf8') as fout:
    fout.writelines(res)
    
#with open('../data/{0}/{0}.hter'.format(fname), 'w', encoding='utf8') as fout:
#    fout.writelines(tgt)