# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
# A Closer Look at Few-shot Classification. ICLR (Poster) 2019

from utils.networks.closer import distLinear
import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate


class Baseline(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, num_class, loss_type="softmax", device='cuda:0'):
        super(Baseline, self).__init__(model_func, n_way, n_support, device=device)
        self.feature_extractor = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature_extractor.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = distLinear(self.feature_extractor.final_feat_dim, num_class)
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, x):
        out = self.feature_extractor.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = y.to(self.device)
        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer):
        self.train()
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss

    def test_loop(self, test_loader, return_std=False):
        self.eval()
        # correct = 0
        # count = 0
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            # ---------------------------
            # TODO temporally replaced the call to correct() with the code
            # correct_this, count_this = self.correct(x)
            scores = self.set_forward(x)
            y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == y_query)
            correct_this = float(top1_correct)
            count_this = len(y_query)
            # ---------------------------
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def set_forward(self, x):
        return self.set_forward_adaptation(x)  # Baseline always do adaptation

    def set_forward_adaptation(self, x):
        z_support, z_query = self.parse_feature(x)
        z_support = z_support.reshape(self.n_way * self.n_support, -1).detach()
        z_query = z_query.contiguous().view(-1, z_support.shape[-1]).detach()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = y_support.to(self.device)

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.to(self.device)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(self.device)
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)])
                selected_id = selected_id.to(self.device)
                z_batch = z_support[selected_id.long()]
                y_batch = y_support[selected_id.long()]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch.long())
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores

    def set_forward_loss(self, x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune networks')
