import torch
from torch.autograd import Variable
from math import floor, sqrt
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from seq2seq.evaluator.estimator import QEstimator
from sklearn.metrics import mean_squared_error,mean_absolute_error


use_gpu = torch.cuda.is_available()


def logcosh(pred, true):
    loss = torch.log(torch.cosh(pred-true))
    return torch.sum(loss)


def avgDelta(reference, preds, mulfactor=100.0):
    inputSort = sorted(zip(preds,reference), key=lambda x:x[0],reverse=True)
    ridx = len(inputSort)
    
    maxN = int(ridx/2)
    avgDelta = [0]*(maxN+1)
    AvgDelta = 0
    
    for cN in range(2, maxN+1):
        refValueSort = [0.0]*(cN+1)
        refSum = 0.0
        for i in range(1,cN+1):
            q = int(ridx/cN)
            head = i*q
            if i==cN and head<ridx:
                head = ridx
            for k in range(head):
                refValueSort[i] += inputSort[k][1]*mulfactor
            refValueSort[i] /= head
            if i<cN:
                refSum += refValueSort[i]
        avgDelta[cN] = refSum/(cN-1)-refValueSort[cN]
        AvgDelta += avgDelta[cN]
    if maxN > 1:
        AvgDelta /= (maxN-1)
    else:
        AvgDelta = 0
    return abs(AvgDelta)


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
        outs.append(pred.cpu())
    preds = torch.cat(outs, 0).squeeze()

    avg_delta = avgDelta(label.tolist(), preds.data.tolist())
    print('Dev epoch {0} deltaAvg {1:5f} rmse {2:5f}, mae {3:5f}, spearmanr {4:5f}'.format(e,
                              avg_delta,
                              sqrt(mean_squared_error(label.tolist(), preds.detach().data.tolist())),
                              mean_absolute_error(label.tolist(), preds.detach().data.tolist()),
                              spearmanr(label.tolist(), preds.detach().data.tolist())[0]
                              ))

    return avg_delta


train = torch.load('./data/persistence/train_vectors.pkl')
target = create_target('./data/train/train.hter')

dstrain = TensorDataset(train.data, target)
dltrain = DataLoader(dstrain, batch_size=32, shuffle=True)


model = QEstimator(train.data.shape[-1], 300)
if use_gpu:
    model = model.cuda()
criterion = logcosh
optim = torch.optim.Adadelta(model.parameters(), lr=0.001)


early_stop = 0
max_delta_avg = [0, 0]

for e in range(1, 100):
    model.train()
    for batch_idx, (train,label) in enumerate(dltrain):
        train = Variable(train)
        label = Variable(label.float())

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
            print('epoch {0} deltaAvg {1:5f}'.
                  format(e, avgDelta(label.cpu().data.tolist(), outs.cpu().data.squeeze().tolist())) )

    avg_delta = eval_(model, e)
    if avg_delta > max_delta_avg[0]:
        max_delta_avg[0] = avg_delta
        max_delta_avg[1] = e
        early_stop = 0
    else:
        early_stop += 1
        if early_stop > 10:
            print('early stop at {0} epoch, average delta {1}'.foramt(max_delta_avg[1], max_delta_avg[0]))
