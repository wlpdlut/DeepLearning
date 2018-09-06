import torch
from math import floor
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from seq2seq.evaluator.estimator import QEstimator

use_gpu = torch.cuda.is_available()


def DeltaAvg(reference, prediction):
    data = [(pred, ref) for pred, ref in zip(prediction, reference)]
    data_sorted = sorted(data, key=lambda x: x[0], reverse=True)
    dataLen = len(data_sorted)

    avg = sum([x[1] for x in data_sorted]) / dataLen
    deltaLen = floor(dataLen // 2 + 1)
    deltaAvg = [0] * deltaLen

    for k in range(2, deltaLen):
        for i in range(1, k):
            deltaAvg[k] += sum([x[1] for x in data_sorted[:floor(dataLen * i / k)]])
        deltaAvg[k] = deltaAvg[k] / (k - 1) - avg
    return sum(deltaAvg) / (deltaLen - 2)


def create_target(fname):
    with open(fname, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
    lines = [float(line.strip('\n')) for line in lines]
    return torch.FloatTensor(lines)


def eval_(model, e):
    model.eval()
    dev = torch.load('./data/persistence/dev_vectors.pkl')
    label = create_target('./data/dev/dev.hter')

    dsdev = TensorDataset(dev.data, label)
    dldev = DataLoader(dsdev, batch_size=32, shuffle=False)

    outs = []
    for batch_idx, (dev, _) in enumerate(dldev):
        dev = torch.autograd.Variable(dev)
        if use_gpu:
            dev = dev.cuda()
        pred = model(dev)
        outs.append(pred)
    preds = torch.cat(outs, 0)
    print('Dev epoch {0} deltaAvg {1:5f} spearman {2:5f}'.format(e,
                              DeltaAvg(label.tolist(), preds.data.squeeze().tolist()),
                              spearmanr(label.tolist(), preds.data.squeeze().tolist())[0]
                              ))


train = torch.load('./data/persistence/train_vectors.pkl')
target = create_target('./data/train/train.hter')

dstrain = TensorDataset(train.data, target)
dltrain = DataLoader(dstrain, batch_size=32, shuffle=True)


model = QEstimator(train.data.shape[-1], 100)
if use_gpu:
    model = model.cuda()

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.05)

for e in range(1, 20):
    model.train()
    for batch_idx, (train,label) in enumerate(dltrain):
        train = torch.autograd.Variable(train)
        label = torch.autograd.Variable(label.float())

        if use_gpu:
            train = train.cuda()
            label = label.cuda()

        optim.zero_grad()
        outs = model(train)

        loss = criterion(outs, label.view(label.size(0), -1))
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(),5)
        optim.step()

        if batch_idx!=0 and batch_idx%50 == 0:
            print('epoch {0} deltaAvg {1:5f} spearman {2:5f}'.
                  format(e,
                         DeltaAvg(label.data.tolist(), outs.data.squeeze().tolist()),
                         spearmanr(label.data.tolist(), outs.data.squeeze().tolist())[0]))
    eval_(model, e)
