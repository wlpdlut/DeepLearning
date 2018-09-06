import torch
from seq2seq.evaluator.vector import Vector

src_vocab = torch.load('./data/persistence/src_vocab.pkl')
tgt_vocab = torch.load('./data/persistence/tgt_vocab.pkl')
model = torch.load('./data/persistence/seq2seq_model.pkl')

file_name = 'dev'

lines = open('./data/{0}/{0}.tsv'.format(file_name), 'r', encoding='utf8').readlines()
vec = Vector(model, src_vocab, tgt_vocab)
res = vec.predict(lines)
print(res.shape)

torch.save(res, './data/persistence/{0}_vectors.pkl'.format(file_name))
