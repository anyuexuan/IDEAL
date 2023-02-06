# From Instance to Class Calibration: A Unified Framework for Open-World Few-Shot Learning. IEEE TPAMI 2023
# This code is modified from https://github.com/anyuexuan/CSS

import torch
from torch import nn
import numpy as np
from methods.meta_template import MetaTemplate
from torchvision import transforms


class ConditionalSimSiam(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, image_size, device='cuda:0'):
        super(ConditionalSimSiam, self).__init__(model_func, n_way, n_support, device=device)
        self.image_size = image_size
        self.projection_mlp_1 = nn.Sequential(
            nn.Linear(self.feature_extractor.final_feat_dim, 2048),
        )
        self.projection_mlp_2 = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048)
        )
        self.prediction_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
        )
        self.to(self.device)

    def f(self, x):
        # x:[N*(S+Q),n_channel,h,w]
        x = self.feature_extractor(x)
        x = self.projection_mlp_1(x)
        x = self.projection_mlp_2(x)
        return x

    def h(self, x):
        # x:[N*(S+Q),2048]
        x = self.prediction_mlp(x)
        return x

    def D(self, p, z):
        z = z.detach()
        p = torch.nn.functional.normalize(p, dim=1)
        z = torch.nn.functional.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def data_augmentation(self, img):
        # x:[n_channel,h,w], torch.Tensor
        x = transforms.RandomResizedCrop(self.image_size)(img)
        x = transforms.RandomHorizontalFlip()(x)
        if np.random.random() < 0.8:
            x = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(x)
        else:
            x = transforms.RandomGrayscale(p=1.0)(x)
        x = transforms.GaussianBlur((5, 5))(x)
        return x

    def contrastive_loss(self, x):
        # x:[N*(S+Q),n_channel,h,w]
        x1 = x.clone()
        x2 = x.clone()
        for index in range(x.shape[0]):
            x1[index] = self.data_augmentation(x[index])
            x2[index] = self.data_augmentation(x[index])
        z1, z2 = self.f(x1), self.f(x2)
        p1, p2 = self.h(z1), self.h(z2)
        loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return loss

    def train_loop(self, epoch, train_loader, optimizer, sl_model=None):
        self.train()
        assert sl_model is not None
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):  # x:[N, S+Q, n_channel, h, w]
            x = x.to(self.device)
            x = x.reshape([x.shape[0] * x.shape[1], *x.shape[2:]])  # x:[N*(S+Q),n_channel,h,w]
            x_ssl = torch.nn.functional.normalize(self.feature_extractor(x), dim=1)
            x_pre = torch.nn.functional.normalize(sl_model.feature_extractor(x).detach(), dim=1)
            loss = self.contrastive_loss(x) - torch.mean(torch.sum((x_ssl * x_pre), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))

    def test_loop(self, test_loader, record=None, return_std=False, image_files=None, image_labels=None):
        self.eval()
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            with torch.no_grad():
                scores = self.set_forward(x)
                y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
                topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
                topk_ind = topk_labels.cpu().numpy()  # index of topk
                acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            print('%d Test Acc = %4.2f±%4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def set_forward(self, x):
        x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.feature_extractor.forward(x)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]

        z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        scores = self.cosine_similarity(z_query, z_proto)
        return scores

    def set_forward_loss(self, x):
        pass
