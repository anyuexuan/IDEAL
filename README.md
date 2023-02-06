# Code for IDEAL

Yuexuan An, Hui Xue, Xingyu Zhao, Jing Wang. From Instance to Metric Calibration: A Unified Framework for Open-world Few-shot Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

## Requirements

- Python >= 3.6
- PyTorch (GPU version) >= 1.5
- NumPy >= 1.13.3
- Scikit-learn >= 0.20

## Getting started

### CIFAR-FS

- Change directory to `./filelists/cifar`
- Download [CIFAR-FS](https://drive.google.com/file/d/1i4atwczSI9NormW5SynaHa1iVN1IaOcs/view)
- run `python make.py` in the terminal

### FC100

- Change directory to `./filelists/fc100`
- Download [FC100](https://drive.google.com/file/d/1jWbj03Fo0SXhd_egH52-rVSP9pUU0dBJ/view)
- run `python make.py` in the terminal

### miniImagenet

- Change directory to `./filelists/miniImagenet`
- Download [miniImagenet](https://drive.google.com/file/d/1hQqDL16HTWv9Jz15SwYh3qq1E4F72UDC/view)
- run `python make.py` in the terminal

### tieredImagenet

- Change directory to `./filelists/tieredImagenet`
- Download [tieredImagenet](https://drive.google.com/file/d/1ir7coqTzg_titf3nrH1brahG2PhuCnpJ/view)
- run `python make.py` in the terminal

## Running the scripts

To pre-train the contrastive network in the terminal, use:

```bash
$ python run_IDEAL_pre_train.py --dataset cifar --model_name Conv4 --train_n_way 5 --test_n_way 5 --n_shot 5 --device cuda:0
```

To train and test the IDEAL model in the terminal, use:

```bash
$ python run_IDEAL.py --dataset cifar --noises 1 --noise_type IT --model_name Conv4 --train_n_way 5 --test_n_way 5 --n_shot 5 --device cuda:0 --meta_algorithm IDEAL --attention_method bilstm --eta 0.1 --gamma 0.1
```

## Acknowledgment

Our project references the codes and datasets in the following repo and papers.

[CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)

Luca Bertinetto, João F. Henriques, Philip H. S. Torr, Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. ICLR 2019.

Boris N. Oreshkin, Pau Rodríguez López, Alexandre Lacoste. TADAM: Task dependent adaptive metric for improved few-shot learning. NeurIPS 2018: 719-729.

Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, Daan Wierstra. Matching Networks for One Shot Learning. NIPS 2016: 3630-3638.

Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richard S. Zemel. Meta-Learning for Semi-Supervised Few-Shot Classification. ICLR 2018.
