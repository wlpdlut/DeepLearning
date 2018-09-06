import torch
from torch.autograd import Variable



class Maxout(torch.nn.Module):
    def __init__(self, pool_size):
        super(Maxout, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        assert(x.shape[-1] % self.pool_size == 0)
        m, _ = x.view(*x.shape[:-1], x.shape[-1]//self.pool_size, self.pool_size).max(-1)
        return m


class Vector(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        self.gpu = torch.cuda.is_available()

        if self.gpu:
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batch_size = 64
        self.maxout = Maxout(55)
        self.pad = 1

    def predict(self, data):
        res = []
        for cur_idx in range(0, len(data), self.batch_size):
            src_seqs, src_lens = [], []
            tgt_seqs = []
            for line in data[cur_idx:cur_idx + self.batch_size]:
                src_line, tgt_line = line.split('\t')
                src_seq, tgt_seq = src_line.strip().split()[:51], tgt_line.strip().split()[:51]

                src_id_seq = [self.src_vocab.stoi[tok] for tok in src_seq]
                tgt_id_seq = [self.tgt_vocab.stoi[tok] for tok in tgt_seq]

                src_seqs.append(src_id_seq)
                src_lens.append(len(src_seq))

                tgt_seqs.append(tgt_id_seq)

            max_seq_length = 51
            max_tgt_length = 51
            [seq.extend([self.pad] * (max_seq_length - len(seq))) for seq in src_seqs]
            [seq.extend([self.pad] * (max_tgt_length - len(seq))) for seq in tgt_seqs]

            _, idx_sort = torch.sort(torch.LongTensor(src_lens), descending=True)
            _, idx_unsort = torch.sort(idx_sort)

            input_var = Variable(torch.LongTensor(src_seqs).index_select(0, idx_sort), volatile=True)
            targets = Variable(torch.LongTensor(tgt_seqs).index_select(0, idx_sort), volatile=True)

            idx_unsort = Variable(idx_unsort)
            if self.gpu:
                input_var = input_var.cuda()
                targets = targets.cuda()

            lengths = torch.LongTensor(src_lens).index_select(0, idx_sort).tolist()
            _, _, _, hhats = self.model(input_var, lengths, targets, teacher_forcing_ratio=1.0)
            hhats = [hhat.cpu().unsqueeze(1) for hhat in hhats]
            hhat = self.maxout(torch.cat(hhats, 1).index_select(0, idx_unsort))

            res.append(hhat)
            if cur_idx % 128 == 0:
                print(cur_idx)
        return torch.cat(res, 0)


