# From Instance to Class Calibration: A Unified Framework for Open-World Few-Shot Learning. IEEE TPAMI 2023

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from torchvision.transforms import Resize, CenterCrop, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize
from PIL import ImageEnhance, Image
import copy


class NoiseMetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, outlier=1, verbose=False, image_size=None, image_files=None,
                 image_labels=None, noise_type='IT', noise_rate=0.5, device='cuda:0'):
        super(NoiseMetaTemplate, self).__init__()
        self.n_way = n_way  # N, n_classes
        self.n_support = n_support  # S, sample num of support set
        self.n_query = -1  # Q, sample num of query set(change depends on input)
        self.feature_extractor = model_func()  # feature extractor
        self.feat_dim = self.feature_extractor.final_feat_dim
        self.verbose = verbose
        self.noise_type = noise_type
        assert self.noise_type in ['IT', 'OOT', 'OOD']
        self.outlier = outlier
        self.noise_rate = noise_rate
        self.image_files = np.array(image_files)
        self.image_labels = image_labels
        self.image_size = image_size
        self.device = device

    @abstractmethod
    def set_forward(self, x):
        # x -> predicted score
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        # x -> loss value
        pass

    def forward(self, x):
        # x-> feature embedding
        out = self.feature_extractor.forward(x)
        return out

    def parse_feature(self, x):
        x = x.requires_grad_(True)
        x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.feature_extractor.forward(x)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]
        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
        topk_ind = topk_labels.cpu().numpy()  # index of topk
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, image_files=None, image_labels=None):
        self.train()
        if image_files is not None:
            self.image_files = image_files
            self.image_labels = image_labels
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if self.image_size is None: self.image_size = x.shape[-2:]
            x = self.add_noise(x, y, noise_type=self.noise_type, aug=True)
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
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
            correct_this, count_this = self.correct(x)
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

    def add_noise(self, x, y, noise_type, aug, return_y=False):
        y = np.array(list(set(y.cpu().numpy().ravel())))
        y_new = torch.zeros([x.shape[0], x.shape[1]]).int().to(self.device)
        for ii in range(x.shape[0]):
            y_new[ii] = y[ii]
        noise_idxes = []
        for i in range(self.n_way):
            noise_idxes.append(np.random.choice(range(self.n_support), size=self.outlier, replace=False))
        noise_idxes = np.array(noise_idxes)
        if noise_type == 'IT':
            temp_x = []
            temp_y = []
            for jj in range(self.outlier):
                temp_x.append(copy.deepcopy(x[0, noise_idxes[0, jj]]))
                temp_y.append(copy.deepcopy(y_new[0, noise_idxes[0, jj]]))
            for ii in range(x.shape[0] - 1):
                for jj in range(self.outlier):
                    x[ii, noise_idxes[ii, jj]] = x[ii + 1, noise_idxes[ii + 1, jj]]
                    y_new[ii, noise_idxes[ii, jj]] = y_new[ii + 1, noise_idxes[ii + 1, jj]]
            for jj in range(self.outlier):
                x[-1, noise_idxes[-1, jj]] = temp_x[jj]
                y_new[-1, noise_idxes[-1, jj]] = temp_y[jj]
        elif noise_type == 'OOT':
            label_mask = np.ones_like(self.image_labels, dtype=int)
            for yy in y:
                label_mask[self.image_labels == yy] = 0
            images_range = np.arange(len(self.image_files))
            files = images_range[label_mask == 1]
            for ii in range(x.shape[0]):
                for jj in range(self.outlier):
                    file = np.random.choice(files)
                    y_new[ii, noise_idxes[ii, jj]] = file
                    xx = Image.open(self.image_files[file]).convert('RGB')
                    if aug:
                        xx = RandomResizedCrop(self.image_size)(xx)
                        xx = ImageJitter()(xx)
                        xx = RandomHorizontalFlip()(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(xx)
                        x[ii, noise_idxes[ii, jj]] = xx
                    else:
                        xx = Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)])(xx)
                        xx = CenterCrop(self.image_size)(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(xx)
                        x[ii, noise_idxes[ii, jj]] = xx
        elif noise_type == 'OOD':
            for ii in range(x.shape[0]):
                for jj in range(self.outlier):
                    file = np.random.choice(self.image_files)
                    xx = Image.open(file).convert('RGB')
                    if aug:
                        xx = RandomResizedCrop(self.image_size)(xx)
                        xx = ImageJitter()(xx)
                        xx = RandomHorizontalFlip()(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(xx)
                        x[ii, noise_idxes[ii, jj]] = xx
                    else:
                        xx = Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)])(xx)
                        xx = CenterCrop(self.image_size)(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(xx)
                        x[ii, noise_idxes[ii, jj]] = xx
        else:
            raise ValueError('Error noise_type!')
        if return_y:
            return x, y_new, noise_idxes
        else:
            return x

    def cosine_similarity(self, x, y):
        '''
        Cosine Similarity of two tensors
        Args:
            x: torch.Tensor, m x d
            y: torch.Tensor, n x d
        Returns:
            result, m x n
        '''
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return x @ y.transpose(0, 1)

    def mahalanobis_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        cov = torch.cov(x)  # [m,m]
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        delta = x - y  # [m,n,d]
        return torch.einsum('abc,abc->ab', torch.einsum('abc,ad->abc', delta, torch.inverse(cov)), delta)

    def euclidean_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return torch.pow(x - y, 2).sum(2)


class ImageJitter(object):
    def __init__(self, transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                                              Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color),
                 transformdict=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
