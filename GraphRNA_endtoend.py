import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix
from sklearn.metrics import f1_score
from math import ceil
from random_walk import walk_dic_featwalk
from models import pro_lstm_featwalk

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=200, help='Number of hidden units.')
parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_paths', type=int, default=100)
parser.add_argument('--alpha', type=float, default=.4)
parser.add_argument('--path_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=10)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def train():
    train_num = idx_train.shape[0]
    splitnum = int(ceil(float(train_num) / args.batch_size))
    model.train()
    np.random.shuffle(idx_train)
    preds = []
    for batch_idx in range(splitnum):
        optimizer.zero_grad()
        indexblock = args.batch_size * batch_idx
        batch_node_idx = idx_train[range(indexblock, indexblock + min(train_num - indexblock, args.batch_size))]

        sentences = []
        for i in batch_node_idx:
            sentences.extend(sentencedic[i])
        outi = model(features[sentences])

        loss = criterion(outi, labels[batch_node_idx])

        preds.append(outi.max(1)[1])

        loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()


def eval(idx_input):
    eval_num = len(idx_input)
    splitnum_eval = int(ceil(float(eval_num) / args.batch_size))
    model.eval()
    preds = []
    for batch_idx in range(splitnum_eval):
        indexblock = args.batch_size * batch_idx
        batch_node_idx = idx_input[range(indexblock, indexblock + min(eval_num - indexblock, args.batch_size))]
        sentences = []
        for i in batch_node_idx:
            sentences.extend(sentencedic[i])
        preds.append(model(features[sentences]).max(1)[1])
    preds = torch.cat((torch.stack(preds[:-1]).view(-1), preds[-1]), 0).cpu()
    labeleval = labels[idx_input].cpu()

    return f1_score(labeleval, preds, average='micro'), f1_score(labeleval, preds, average='macro')


if __name__ == "__main__":
    '''################# Experimental Settings #################'''
    mat_contents = sio.loadmat('data/Flickr_SDM.mat')
    A = mat_contents["Attributes"]
    Label = mat_contents["Label"]
    G = mat_contents["Network"]
    G.setdiag(0)
    n, m = A.shape  # num of nodes

    Indices = np.random.randint(25, size=n) + 1
    Group2 = []
    [Group2.append(x) for x in range(0, len(Indices)) if Indices[x] >= 21]  # test group
    n2 = len(Group2)  # num of nodes in test group
    Group1 = []
    [Group1.append(x) for x in range(0, len(Indices)) if Indices[x] <= 20]
    n1 = len(Group1)  # num of nodes in training group

    features = A[Group1 + Group2, :]  # torch.tensor(A[Group1+Group2, :].todense(), dtype=torch.float)
    labels = torch.tensor(Label[Group1 + Group2, 0] - 1, dtype=torch.long)
    adj = G[Group1 + Group2, :][:, Group1 + Group2].todense()

    idx_test = torch.arange(n1, n1 + n2)
    idx_train, idx_val = train_test_split(range(n1), test_size=0.1)
    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)

    start_time = time.time()
    adj = csc_matrix(adj)

    sentencedic, sentnumdic = walk_dic_featwalk(adj, features, num_paths=args.num_paths,
                                                path_length=args.path_length, alpha=args.alpha).function()

    model = pro_lstm_featwalk(nfeat=features.shape[1],  # num of categories
                              nhid=args.hidden,
                              nclass=labels.max().item() + 1,
                              dropout=args.dropout,
                              num_paths=args.num_paths,
                              path_length=args.path_length)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    features = torch.tensor(features.todense(), dtype=torch.float)
    features = torch.cat((features, torch.eye(features.shape[1])), 0)

    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
        labels = labels.cuda()
        features = features.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    criterion = nn.CrossEntropyLoss()
    # Train model
    early_stop = False
    best_dev_acc = 0
    iters_not_improved = 0
    patience = args.patience  # for early stopping

    for epoch in range(args.epochs):
        if early_stop:
            print("Early Stopping. Epoch: {}, Best Dev Recall: {}".format(epoch, best_dev_acc))
            break

        train()
        val_acc, __ = eval(idx_val)
        if val_acc >= best_dev_acc:
            best_dev_acc = val_acc
            iters_not_improved = 0
            imicro, imacro = eval(idx_test)

        else:
            iters_not_improved += 1
            if iters_not_improved > patience:
                early_stop = True
                break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))
    print("Test set results:", "accuracy= {:.4f}".format(imicro))




