from utils.utils import *
from methods.conditionasimsiam import ConditionalSimSiam
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar',
                    choices=['cifar', 'fc100', 'miniImagenet', 'tieredImagenet'])
parser.add_argument('--model_name', type=str, default='Conv4',
                    choices=['Conv4', 'ResNet12'])
parser.add_argument('--train_n_way', type=int, default=5)
parser.add_argument('--test_n_way', type=int, default=5)
parser.add_argument('--n_shot', type=int, default=5)
parser.add_argument('--stop_epoch', type=int, default=-1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--pre_algorithm', type=str, default='baseline++')
parser.add_argument('--ssl_algorithm', type=str, default='conditionalsimsiam')


def pre_train():
    model = get_model(algorithm=pre_algorithm, model_name=model_name, dataset=dataset, n_way=train_n_way,
                      n_shot=n_shot, adaptation=adaptation, )
    model = model.cuda()
    optimizer = get_optimizer(model, model_name)
    for epoch in range(start_epoch, pre_epoch):
        model.train()
        model.train_loop(epoch, pre_base_loader, optimizer)  # model are called by reference, no need to return
        if epoch == pre_epoch - 1:
            outfile = os.path.join(pre_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)


def pre_test():
    model = get_model(algorithm=pre_algorithm, model_name=model_name, dataset=dataset, n_way=train_n_way,
                      n_shot=n_shot, adaptation=adaptation, )
    modelfile = get_resume_file(pre_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])
    model.eval()
    acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
    print('%d Test Acc = %4.2f±%4.2f%%' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))


def ssl_train():
    sl_model = get_model(algorithm=pre_algorithm, model_name=model_name, dataset=dataset, n_way=train_n_way,
                         n_shot=n_shot, adaptation=adaptation, )
    modelfile = get_resume_file(pre_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    sl_model.load_state_dict(tmp['state'])
    sl_model.eval()
    model = ConditionalSimSiam(model_func=model_dict[model_name], n_way=train_n_way, n_support=n_shot,
                               image_size=image_size, )
    optimizer = get_optimizer(model, model_name)
    max_acc = 0
    for epoch in range(start_epoch, ssl_epoch):
        model.train()
        model.train_loop(epoch, ssl_base_loader, optimizer,
                         sl_model=sl_model)  # model are called by reference, no need to return
        if epoch == ssl_epoch - 1:
            outfile = os.path.join(ssl_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        model.eval()
        acc = model.test_loop(ssl_val_loader)
        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print("--> Best model! save...", acc)
            max_acc = acc
            outfile = os.path.join(ssl_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)


def ssl_test():
    model = ConditionalSimSiam(model_func=model_dict[model_name], n_way=train_n_way, n_support=n_shot,
                               image_size=image_size, )
    modelfile = get_best_file(ssl_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])
    model.eval()
    acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
    print('%d Test Acc = %4.2f±%4.2f%%' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model_name
    train_n_way = args.train_n_way
    test_n_way = args.test_n_way
    n_shot = args.n_shot
    stop_epoch = args.stop_epoch
    device = args.device
    pre_algorithm = args.pre_algorithm
    ssl_algorithm = args.ssl_algorithm

    image_size = get_image_size(model_name=model_name, dataset=dataset)
    model_name = get_model_name(model_name=model_name, dataset=dataset)

    if stop_epoch == -1:
        pre_epoch = get_stop_epoch(algorithm=pre_algorithm, dataset=dataset, n_shot=n_shot)
        ssl_epoch = get_stop_epoch(algorithm=ssl_algorithm, dataset=dataset, n_shot=n_shot)
    else:
        pre_epoch = stop_epoch
        ssl_epoch = stop_epoch

    pre_dir = base_path + f'/save/checkpoints/{dataset}/{model_name}_{pre_algorithm}'
    ssl_dir = base_path + f'/save/checkpoints/{dataset}/{model_name}_{pre_algorithm}_{ssl_algorithm}'

    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(ssl_dir, exist_ok=True)

    base_file, val_file = get_train_files(dataset=dataset)
    pre_base_loader, pre_val_loader = get_train_loader(algorithm=pre_algorithm, image_size=image_size,
                                                       base_file=base_file, val_file=val_file,
                                                       train_n_way=train_n_way,
                                                       test_n_way=test_n_way, n_shot=n_shot,
                                                       num_workers=num_workers)
    ssl_base_loader, ssl_val_loader = get_train_loader(algorithm=ssl_algorithm, image_size=image_size,
                                                       base_file=base_file, val_file=val_file,
                                                       train_n_way=train_n_way,
                                                       test_n_way=test_n_way, n_shot=n_shot,
                                                       num_workers=num_workers)
    datamgr = SetDataManager(image_size, n_eposide=test_iter_num, n_query=15, n_way=test_n_way,
                             n_support=n_shot, num_workers=num_workers)
    novel_file = get_novel_file(dataset=dataset, split='novel')
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    # ----------------------------------------Pre Training-----------------------------------------
    print(dataset, pre_algorithm, model_name, 'Start pre-training!')
    if os.path.exists(os.path.join(pre_dir, '{:d}.tar'.format(pre_epoch - 1))):
        print('Using exist model!')
    else:
        pre_train()
    # ----------------------------------------SSL Training-----------------------------------------
    print(dataset, ssl_algorithm, model_name, 'Start ssl-training!')
    if os.path.exists(os.path.join(ssl_dir, '{:d}.tar'.format(ssl_epoch - 1))):
        print('Using exist model!')
    else:
        ssl_train()
