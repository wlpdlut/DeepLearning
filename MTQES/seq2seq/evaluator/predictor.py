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


class Predictor(object):

    def __init__(self, model, src_vocab):
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

    def predict_batch(self, data, batch_size=64):
        maxout = Maxout(55)
        pad = 1
        res = []
        for cur_idx in range(0, len(data), batch_size):
            seqs = []
            lens = []
            for line in data[cur_idx:cur_idx+batch_size]:
                src_seq = line.strip().split()
                src_id_seq = [self.src_vocab.stoi[tok] for tok in src_seq]
                seqs.append(src_id_seq)
                lens.append(len(src_seq))

            max_seq_length = max(lens)
            [seq.extend([pad]*(max_seq_length-len(seq))) for seq in seqs]

            _, idx_sort = torch.sort(torch.LongTensor(lens), descending=True)
            _, idx_unsort = torch.sort(idx_sort)

            input_var = Variable(torch.LongTensor(seqs).index_select(0, idx_sort), volatile=True)
            idx_unsort = Variable(idx_unsort)
            if self.gpu:
                input_var = input_var.cuda()
                #idx_unsort = idx_unsort.cuda()
            lengths = torch.LongTensor(lens).index_select(0, idx_sort).tolist()
            _, _, _, hhats = self.model(input_var, lengths)
            hhats = [hhat.cpu().unsqueeze(1) for hhat in hhats]
            hhat = maxout(torch.cat(hhats, 1).index_select(0, idx_unsort))

            res.append(hhat)
            if cur_idx % 128 == 0:
                print(cur_idx)
        return torch.cat(res, 0)

    def predict(self, data):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        iLoop = 1
        res = []
        maxout = Maxout(55)
        for line in data:
            src_seq = line.strip().split()
            src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                                                    volatile=True).view(1, -1)
            if self.gpu:
                src_id_seq = src_id_seq.cuda()

            _, _, _, hhat = self.model(src_id_seq, [len(src_seq)])
            hhat = torch.cat(hhat, 0)
            hhat = maxout(hhat).unsqueeze(0)
            res.append(hhat)

            if iLoop % 100 == 0:
                print(iLoop)
            iLoop += 1

        return torch.cat(res, 0)
