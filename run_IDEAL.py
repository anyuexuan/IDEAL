import os
from utils.utils import *
from methods.conditionasimsiam import ConditionalSimSiam
from methods.IDEAL import IDEAL
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar',
                    choices=['cifar', 'fc100', 'miniImagenet', 'tieredImagenet'])
parser.add_argument('--noises', type=int, default=1,
                    help='number of noise samples')
parser.add_argument('--noise_type', type=str, default='IT',
                    choices=['IT', 'OOT', 'OOD'])
parser.add_argument('--model_name', type=str, default='Conv4',
                    choices=['Conv4', 'ResNet12'])
parser.add_argument('--train_n_way', type=int, default=5)
parser.add_argument('--test_n_way', type=int, default=5)
parser.add_argument('--n_shot', type=int, default=5)
parser.add_argument('--stop_epoch', type=int, default=-1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--pre_algorithm', type=str, default='baseline++')
parser.add_argument('--ssl_algorithm', type=str, default='conditionalsimsiam')
parser.add_argument('--meta_algorithm', type=str, default='IDEAL')
parser.add_argument('--attention_method', type=str, default='bilstm',
                    choices=['bilstm', 'transformer'])
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)


def meta_train():
    if noise_type == 'OOD':
        raise ValueError("OOD don't support training!")
    ssl_model = ConditionalSimSiam(model_func=model_dict[model_name], n_way=train_n_way, n_support=n_shot,
                                   image_size=image_size, device=device)
    modelfile = get_best_file(ssl_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    ssl_model.load_state_dict(tmp['state'])
    ssl_model.eval()
    model = IDEAL(model_func=model_dict[model_name], n_way=train_n_way, n_support=n_shot, image_size=image_size,
                  outlier=outlier, image_files=train_files, image_labels=train_labels, noise_type=noise_type,
                  ssl_feature_extractor=ssl_model.feature_extractor, attention_method=attention_method,
                  eta=eta, gamma=gamma, device=device)
    optimizer = get_optimizer(model, model_name)
    max_acc = 0
    for epoch in range(start_epoch, meta_epoch):
        model.train()
        model.train_loop(epoch, meta_base_loader, optimizer, image_files=train_files,
                         image_labels=train_labels)  # model are called by reference, no need to return
        model.eval()
        acc = model.test_loop(meta_val_loader, image_files=val_files, image_labels=val_labels)
        if acc > max_acc:
            print(acc, "--> Best model! save...")
            max_acc = acc
            outfile = os.path.join(meta_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        if epoch == meta_epoch - 1:
            outfile = os.path.join(meta_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)


def meta_test():
    ssl_model = ConditionalSimSiam(model_func=model_dict[model_name], n_way=train_n_way, n_support=n_shot,
                                   image_size=image_size, device=device)
    modelfile = get_best_file(ssl_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    ssl_model.load_state_dict(tmp['state'])
    ssl_model.eval()
    model = IDEAL(model_func=model_dict[model_name], n_way=train_n_way, n_support=n_shot, image_size=image_size,
                  outlier=outlier, image_files=train_files, image_labels=train_labels, noise_type=noise_type,
                  ssl_feature_extractor=ssl_model.feature_extractor, attention_method=attention_method,
                  eta=eta, gamma=gamma, device=device)
    modelfile = get_best_file(meta_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])
    model.eval()
    acc_mean, acc_std = model.test_loop(novel_loader, return_std=True, image_files=novel_files,
                                        image_labels=novel_labels)
    print('%d Test Acc = %4.2f±%4.2f%%' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    outlier = args.noises
    noise_type = args.noise_type
    model_name = args.model_name
    train_n_way = args.train_n_way
    test_n_way = args.test_n_way
    n_shot = args.n_shot
    stop_epoch = args.stop_epoch
    device = args.device
    pre_algorithm = args.pre_algorithm
    ssl_algorithm = args.ssl_algorithm
    meta_algorithm = args.meta_algorithm
    attention_method = args.attention_method
    eta = args.eta
    gamma = args.gamma

    image_size = get_image_size(model_name=model_name, dataset=dataset)
    model_name = get_model_name(model_name=model_name, dataset=dataset)

    if stop_epoch == -1:
        meta_epoch = get_stop_epoch(algorithm=meta_algorithm, dataset=dataset, n_shot=n_shot)
    else:
        meta_epoch = stop_epoch

    ssl_dir = base_path + f'/save/checkpoints/{dataset}/{model_name}_{pre_algorithm}_{ssl_algorithm}'
    base_file, val_file = get_train_files(dataset=dataset)
    meta_base_loader, meta_val_loader = get_train_loader(algorithm=meta_algorithm, image_size=image_size,
                                                         base_file=base_file, val_file=val_file,
                                                         train_n_way=train_n_way,
                                                         test_n_way=test_n_way, n_shot=n_shot,
                                                         num_workers=num_workers)

    train_labels = json.load(open(base_file))['image_labels']
    train_files = json.load(open(base_file))['image_names']
    val_labels = json.load(open(val_file))['image_labels']
    val_files = json.load(open(val_file))['image_names']

    datamgr = SetDataManager(image_size, n_eposide=test_iter_num, n_query=15, n_way=test_n_way,
                             n_support=n_shot, num_workers=num_workers)
    novel_file = get_novel_file(dataset=dataset, split='novel')
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    if noise_type in ['IT', 'OOT']:
        outlier_file = novel_file
    elif noise_type == 'OOD':
        datamgr = SetDataManager(image_size, n_eposide=test_iter_num, n_query=15, n_way=test_n_way,
                                 n_support=n_shot, num_workers=num_workers)
        if dataset in ['cifar', 'fc100', 'CUB']:
            outlier_file = get_novel_file(dataset='miniImagenet', split='novel')
        elif dataset in ['miniImagenet', 'tieredImagenet']:
            outlier_file = get_novel_file(dataset='cifar', split='novel')
        else:
            raise ValueError('Wrong dataset name!')
    else:
        raise ValueError('Noise Type Error!')
    novel_labels = json.load(open(outlier_file))['image_labels']
    novel_files = json.load(open(outlier_file))['image_names']

    # ----------------------------------------Meta Training-----------------------------------------
    print(dataset, meta_algorithm, attention_method, model_name, train_n_way, n_shot, noise_type, outlier, eta, gamma,
          'Start meta-training!')
    if noise_type == 'OOD':
        meta_dir = base_path + f'/save/checkpoints/{dataset}/{model_name}_{pre_algorithm}_{ssl_algorithm}_{meta_algorithm}_OOT_{outlier}_{attention_method}_{eta}_{gamma}'
    else:
        meta_dir = base_path + f'/save/checkpoints/{dataset}/{model_name}_{pre_algorithm}_{ssl_algorithm}_{meta_algorithm}_{noise_type}_{outlier}_{attention_method}_{eta}_{gamma}'
    os.makedirs(meta_dir, exist_ok=True)
    if os.path.exists(os.path.join(meta_dir, '{:d}.tar'.format(meta_epoch - 1))):
        print('Using exist model!')
    else:
        meta_train()
    print(dataset, meta_algorithm, attention_method, model_name, train_n_way, n_shot, noise_type, outlier, eta, gamma,
          'Start meta-testing!')
    meta_test()
