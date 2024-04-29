import os
import numpy as np
import torch
from torchvision import transforms
import sys
import argparse
import multiprocessing as mp

# 假设 datasets.py 和 prepare.py 在同一目录下
sys.path.append(os.path.dirname(__file__))

import sys
# 假设你的.py文件分别位于不同的Google Drive文件夹中
#folder_path_1 = '/content/drive/My Drive/capstone5703/GALIP/code/'
folder_path_2 = '/content/drive/My Drive/capstone5703/GALIP/code/lib'

# 添加这些文件夹到sys.path中
#sys.path.append(folder_path_1)
sys.path.append(folder_path_2)


from datasets import TextImgDataset, get_imgs, get_caption
from perpare import prepare_dataloaders, load_clip, prepare_models

'''
class Args:
    def __init__(self):
        self.data_dir = '/path/to/your/dataset'  # 请根据实际情况修改路径
        self.dataset_name = 'your_dataset_name'  # 根据你的数据集名称进行修改
        self.clip4text = 'ViT-B/32'  # CLIP模型版本
        self.clip4trn = {'type': 'ViT-B/32'}  # 训练时使用的CLIP模型
        self.clip4evl = {'type': 'ViT-B/32'}  # 评估时使用的CLIP模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32  # 可以根据你的GPU内存调整
        self.num_workers = 4  # 根据你的系统配置调整
        self.imsize = 256  # 图像尺寸
        self.multi_gpus = False  # 如果使用多GPU设置为True
        self.gpu_id: 0
        self.nf: 64
        self.ch_size: 3

        # 添加其他可能需要的参数

'''


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/coco.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--pretrained_model_path', type=str, default='model',
                        help='the model for training')
    parser.add_argument('--log_dir', type=str, default='new',
                        help='file path to log directory')
    parser.add_argument('--model', type=str, default='GALIP',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=100,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--multi_gpus', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--imsize', type=int, default=256, help='image size')
    parser.add_argument('--ch_size', type=int, default=3, help='ch_size')
    parser.add_argument('--clip4text', type=str, default='ViT-B/32',
                        help='CLIP model version for text processing')
    parser.add_argument('--clip4trn', type=str, default='ViT-B/32',
                        help='CLIP model version for text processing')
    parser.add_argument('--clip4evl', type=str, default='ViT-B/32',
                        help='CLIP model version for text processing')
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/capstone5703/GALIP/data/birds',
                        help='Directory for dataset.')
    parser.add_argument('--dataset_name', type=str, default='birds',
                        help='Name of the dataset.')
    args = parser.parse_args()
    return args



def compute_statistics(dataloader):
    # 计算数据的均值和标准差
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([0.0, 0.0, 0.0])
    total_images = 0

    for imgs, _, _, _ in dataloader:
        imgs = imgs.numpy()  # 将tensor转换为numpy数组
        batch_samples = imgs.shape[0]
        imgs = imgs.reshape(batch_samples, imgs.shape[1], -1)
        mu += imgs.mean(2).sum(0)
        sigma += imgs.std(2).sum(0)
        total_images += batch_samples

    mu /= total_images
    sigma /= total_images

    return mu, sigma

def main():
    args = parse_args()  # 初始化参数
    # 准备数据加载器
    transform = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.ToTensor(),
    ])
    #train_dataloader, valid_dataloader, _, _ = prepare_dataloaders(args, transform)
    train_dataloader, valid_dataloader, train_dataset, valid_dataset, train_sampler = prepare_dataloaders(args, transform)

    # 计算统计数据
    print("Computing statistics for training data...")
    mu_train, sigma_train = compute_statistics(train_dataloader)
    print("Training Data: Mu: {}, Sigma: {}".format(mu_train, sigma_train))

    print("Computing statistics for validation data...")
    mu_valid, sigma_valid = compute_statistics(valid_dataloader)
    print("Validation Data: Mu: {}, Sigma: {}".format(mu_valid, sigma_valid))

    # 保存为npz文件
    np.savez("/path/to/save/data_stats.npz", mu_train=mu_train, sigma_train=sigma_train, mu_valid=mu_valid, sigma_valid=sigma_valid)
    print("Statistics saved to /path/to/save/data_stats.npz")

if __name__ == "__main__":
    main()
