# From Instance to Class Calibration: A Unified Framework for Open-World Few-Shot Learning. IEEE TPAMI 2023

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from methods.noisemeta_template import NoiseMetaTemplate
from sklearn.cluster import KMeans
import copy


class IDEAL(NoiseMetaTemplate):
    def __init__(self, model_func, n_way, n_support, image_size, outlier, ssl_feature_extractor,
                 image_files=None, image_labels=None,
                 eta=0.1, gamma=0.1, hidden_dim=20, noise_type='IT', attention_method='bilstm', device='cuda:0'):
        super(IDEAL, self).__init__(model_func,
                                    n_way,
                                    n_support,
                                    image_size=image_size,
                                    image_files=image_files,
                                    outlier=outlier,
                                    image_labels=image_labels,
                                    noise_type=noise_type,
                                    device=device)
        self.ssl_feature_extractor = copy.deepcopy(ssl_feature_extractor)
        self.loss_fn = nn.CrossEntropyLoss()
        self.attention_method = attention_method
        if self.attention_method == 'bilstm':
            self.corr_encoder = nn.LSTM(input_size=self.n_support, hidden_size=hidden_dim, num_layers=2,
                                        batch_first=True, bidirectional=True)
            self.norm_encoder = nn.Sequential(
                nn.Linear(2 * hidden_dim * self.n_support, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_support)
            )
        elif self.attention_method == 'transformer':
            self.attentive = nn.Transformer(d_model=self.n_support, nhead=5, dim_feedforward=2 * hidden_dim)
            self.norm_encoder = nn.Sequential(
                nn.Linear(self.n_support * self.n_support, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_support)
            )
        self.eta = eta
        self.gamma = gamma
        self.to(self.device)

    def get_attention_class(self, x):
        # Intra-class Calibration
        xx = x[:, :self.n_support].reshape(self.n_way * self.n_support, *x.size()[2:])
        z_support = self.ssl_feature_extractor.forward(xx).reshape(self.n_way, self.n_support, -1)
        z_support_norm = z_support / torch.norm(z_support, 2, 2).unsqueeze(2)  # [N,S,S]
        correlation = torch.bmm(z_support_norm, z_support_norm.permute(0, 2, 1))  # [N,S,S]
        if self.attention_method == 'bilstm':
            z_out, _ = self.corr_encoder.forward(correlation)  # [N,S,2*hidden_dim]
        elif self.attention_method == 'transformer':
            z_out = self.attentive(correlation, correlation)  # [N,S,S]
        else:
            raise ValueError('Error attention method!')
        c_out = self.norm_encoder.forward(z_out.reshape(self.n_way, -1))  # [N,S]
        attention_class = F.softmax(c_out, dim=1)  # [N,S]
        return attention_class

    def get_attention_support(self, x):
        # Inter-class Calibration
        with torch.no_grad():
            xx = x[:, :self.n_support].reshape(self.n_way * self.n_support, *x.size()[2:])
            z_support = self.ssl_feature_extractor.forward(xx).reshape(self.n_way, self.n_support, -1)
            z_proto = z_support.mean(1)  # [N,d]
            z_support = z_support.reshape(-1, z_support.shape[-1])
            cluster = KMeans(self.n_way, init=z_proto.detach().cpu().numpy(), n_init=1)
            cluster.fit(z_support.cpu().numpy())
            proto = torch.from_numpy(cluster.cluster_centers_).to(self.device)  # [N,d]
        sim = self.cosine_similarity(z_support, proto) * 10  # [N*S,N]
        attention_support = torch.softmax(sim, dim=0)  # [N*S,N]
        return attention_support

    def set_forward(self, x):
        attention_class = self.get_attention_class(x)
        attention_support = self.get_attention_support(x)

        xx = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.ssl_feature_extractor.forward(xx)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]
        z_proto_class = torch.sum(z_support * attention_class.unsqueeze(2), dim=1)  # [N,d]
        z_proto_support = attention_support.transpose(0, 1) @ z_support.reshape(self.n_way * self.n_support, -1)
        z_proto = z_proto_class * 0.9 + z_proto_support * 0.1
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        scores_ssl = self.cosine_similarity(z_query, z_proto) * 10

        z_support, z_query = self.parse_feature(x)
        z_proto_class = torch.sum(z_support * attention_class.unsqueeze(2), dim=1)  # [N,d]
        z_proto_support = attention_support.transpose(0, 1) @ z_support.reshape(self.n_way * self.n_support, -1)
        z_proto = z_proto_class * 0.9 + z_proto_support * 0.1
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        scores_sl = self.cosine_similarity(z_query, z_proto) * 10

        return scores_ssl + scores_sl

    def set_forward_loss(self, x, y=None, noise_idxes=None):
        assert y is not None
        assert noise_idxes is not None
        # ------------------------Meta Loss------------------------
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        y_query = y_query.to(self.device)
        scores = self.set_forward(x)
        loss_classification = self.loss_fn(scores, y_query)

        # ------------------------Intra-class Noise Loss------------------------
        attention_class = self.get_attention_class(x)
        mask = torch.zeros_like(attention_class).to(self.device)
        for i in range(self.n_way):
            mask[i, noise_idxes[i]] = 1
        loss_intra = torch.mean(torch.sum(-torch.log(
            torch.max(1 - attention_class, torch.tensor([1e-14]).to(self.device))) * mask, dim=1))

        # ------------------------Inter-class Noise Loss------------------------
        xx = x[:, :self.n_support].reshape(self.n_way * self.n_support, *x.size()[2:])
        yy = y[:, :self.n_support].reshape(-1)
        z = self.feature_extractor.forward(xx)  # [N*S, d]
        sim = self.cosine_similarity(z, z)
        mask = torch.ones_like(sim).to(self.device)
        for ii in range(z.shape[0]):
            mask[ii, yy[ii] == yy] = 1
            mask[ii, yy[ii] != yy] = -1
            mask[ii, ii] = 0
        loss_inter = -torch.mean(sim * mask)
        loss = loss_classification + loss_intra * self.eta + loss_inter * self.gamma
        return loss

    def train_loop(self, epoch, train_loader, optimizer, image_files=None, image_labels=None):
        self.train()
        if image_files is not None:
            self.image_files = image_files
            self.image_labels = image_labels
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if self.image_size is None: self.image_size = x.shape[-2:]
            x, y, noise_idxes = self.add_noise(x, y, noise_type=self.noise_type, aug=True, return_y=True)
            x = x.to(self.device)
            y = y.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, y, noise_idxes)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss

    def test_loop(self, test_loader, return_std=False, image_files=None, image_labels=None):
        self.eval()
        if image_files is not None:
            self.image_files = image_files
            self.image_labels = image_labels
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            x = self.add_noise(x, y, noise_type=self.noise_type, aug=False)
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            with torch.no_grad():
                scores = self.set_forward(x)
            y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
            topk_ind = topk_labels.cpu().numpy()  # index of topk
            top1_correct = np.sum(topk_ind[:, 0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            # Confidence Interval   90% -> 1.645      95% -> 1.96     99% -> 2.576
            print('%d Test Acc = %4.2f±%4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
