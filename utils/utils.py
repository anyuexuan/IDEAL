import torch
import numpy as np
from .backbone import *
import os
import random
import glob
from methods.baseline import Baseline
from data.datamgr import SimpleDataManager, SetDataManager
import warnings

warnings.filterwarnings("ignore")

base_path = os.path.dirname(__file__).replace('\\', '/') + '/..'

model_dict = dict(
    Conv4=Conv64F,
    ResNet12=resnet12,
)

data_dir = dict(
    CUB=os.getcwd() + '/filelists/CUB/',
    miniImagenet=os.getcwd() + '/filelists/miniImagenet/',
    omniglot=os.getcwd() + '/filelists/omniglot/',
    emnist=os.getcwd() + '/filelists/emnist/',
    cifar=os.getcwd() + '/filelists/cifar/',
    fc100=os.getcwd() + '/filelists/fc100/',
    tieredImagenet=os.getcwd() + '/filelists/tieredImagenet/',
)

start_epoch = 0  # Starting epoch
train_n_way = 5  # class num to classify for training
test_n_way = 5  # class num to classify for testing (validation)
num_workers = 8
test_iter_num = 600
adaptation = False

if torch.cuda.is_available():
    use_cuda = True
    print('GPU detected, running with GPU!')
else:
    print('GPU not detected, running with CPU!')
    use_cuda = False


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 0
set_seed(seed)


def get_stop_epoch(algorithm, dataset, n_shot=5):
    if algorithm in ['baseline', 'baseline++']:
        if dataset in ['omniglot', 'cross_char']:
            stop_epoch = 5
        elif dataset in ['CUB']:
            stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
        else:
            stop_epoch = 400  # default
    else:
        if n_shot == 5:
            stop_epoch = 400
        else:
            stop_epoch = 600  # default
    return stop_epoch


def get_image_size(model_name, dataset):
    if 'Conv' in model_name and dataset in ['omniglot', 'cross_char']:
        image_size = 28
    elif 'Conv' in model_name or model_name == 'ResNet12':
        image_size = 84
    else:
        image_size = 224
    return image_size


def get_train_files(dataset):
    if dataset == 'cross':
        base_file = data_dir['miniImagenet'] + 'all.json'
        val_file = data_dir['CUB'] + 'val.json'
    elif dataset == 'cross_char':
        base_file = data_dir['omniglot'] + 'noLatin.json'
        val_file = data_dir['emnist'] + 'val.json'
    else:
        base_file = data_dir[dataset] + 'base.json'
        val_file = data_dir[dataset] + 'val.json'
    return base_file, val_file


def get_train_loader(algorithm, image_size, base_file, val_file, train_n_way=5, test_n_way=5, n_shot=5,
                     num_workers=num_workers):
    if 'baseline' in algorithm:
        base_datamgr = SimpleDataManager(image_size, batch_size=16, num_workers=num_workers)
        base_loader = base_datamgr.get_data_loader(base_file, aug=True)
        val_datamgr = SimpleDataManager(image_size, batch_size=64, num_workers=num_workers)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else:
        n_query = max(1, int(
            16 * test_n_way / train_n_way))  # if test_n_way <train_n_way, reduce n_query to keep batch size small
        base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=train_n_way, n_support=n_shot,
                                      num_workers=num_workers)  # n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=True)
        val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=test_n_way, n_support=n_shot,
                                     num_workers=num_workers)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
    return base_loader, val_loader


def get_novel_file(dataset, split='novel'):
    if dataset == 'cross':
        if split == 'base':
            loadfile = data_dir['miniImagenet'] + 'all.json'
        else:
            loadfile = data_dir['CUB'] + split + '.json'
    elif dataset == 'cross_char':
        if split == 'base':
            loadfile = data_dir['omniglot'] + 'noLatin.json'
        else:
            loadfile = data_dir['emnist'] + split + '.json'
    else:
        loadfile = data_dir[dataset] + split + '.json'
    return loadfile


def get_model_name(model_name, dataset):
    if dataset in ['omniglot', 'cross_char']:
        assert model_name == 'Conv4', 'omniglot only support Conv4 without augmentation'
        model_name = 'Conv4S'
    return model_name


def get_resume_file(checkpoint_dir, epoch=None):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    if epoch is not None:
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
        return resume_file
    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def get_num_class(dataset):
    if dataset == 'omniglot':
        num_classes = 4112  # total number of classes in softmax, only used in baseline
    elif dataset == 'cross_char':
        num_classes = 1597
    elif dataset in ['cifar', 'fc100', 'miniImagenet', 'cross']:
        num_classes = 100
    elif dataset == 'tieredImagenet':
        num_classes = 608
    else:
        num_classes = 200
    return num_classes


def get_model(algorithm, model_name, dataset, n_way, n_shot, adaptation):
    num_classes = get_num_class(dataset)
    if algorithm == 'baseline':
        model = Baseline(model_dict[model_name], n_way=n_way, n_support=n_shot, num_class=num_classes,
                         loss_type='softmax', )
    elif algorithm == 'baseline++':
        model = Baseline(model_dict[model_name], n_way=n_way, n_support=n_shot, num_class=num_classes,
                         loss_type='dist', )
    else:
        raise ValueError('Unknown algorithm')
    return model


def get_checkpoint_dir(algorithm, model_name, dataset, train_n_way, n_shot, addition=None):
    if dataset == 'cross':
        dataset = 'miniImagenet'
    elif dataset == 'cross_char':
        dataset = 'omniglot'
    if addition is None:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s' % (dataset, model_name, algorithm)
    else:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s_%s' % (dataset, model_name, algorithm, str(addition))
    if not algorithm in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_optimizer(model, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return optimizer


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1).long(), 1)
