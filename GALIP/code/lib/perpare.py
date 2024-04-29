import sys

# 假设你的.py文件分别位于不同的Google Drive文件夹中
#folder_path_1 = '/content/drive/My Drive/capstone5703/GALIP/code/'
folder_path_2 = '/content/drive/My Drive/capstone5703/GALIP/code'

# 添加这些文件夹到sys.path中
#sys.path.append(folder_path_1)
sys.path.append(folder_path_2)



import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import clip
import importlib
from lib.utils import choose_model
from transformers import logging, AlbertConfig, BertConfig, BertModel
from torch.nn.utils.rnn import pad_sequence

###########   preparation   ############
def load_clip(clip_info, device):
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model


def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    CLIP4trn = load_clip(args.clip4trn, device).eval()
    CLIP4evl = load_clip(args.clip4evl, device).eval()
    bert_model_name = args.bert_model_name #or‘bert-base-chinese' directly
    num_labels = args.num_labels 
    config = BertConfig.from_pretrained(bert_model_name, num_labels=num_labels)
    config.output_hidden_states = True
    BERT = BertModel.from_pretrained(bert_model_name, config=config).eval()
    NetG,NetD,NetC,CLIP_IMG_ENCODER,CLIP_TXT_ENCODER = choose_model(args.model)
    # image encoder
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()
    # text encoder
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()
    BERT = BERT.to(device)
    for param in BERT.parameters():
        param.requires_grad = False
    BERT.eval()
    # GAN models
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size, args.mixed_precision, CLIP4trn).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size, args.mixed_precision, args.num_classes).to(device) ## 4.17
    netC = NetC(args.nf, args.cond_dim, args.text_feature_size, args.hidden_size, args.output_size, args.mixed_precision).to(device)##4.12
    #netC = NetC(args.nf, args.cond_dim, args.mixed_precision).to(device)
    if (args.multi_gpus) and (args.train):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = torch.nn.parallel.DistributedDataParallel(netG, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netD = torch.nn.parallel.DistributedDataParallel(netD, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netC = torch.nn.parallel.DistributedDataParallel(netC, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, netG, netD, netC, BERT


def prepare_dataset(args, split, transform):
    if args.ch_size!=3:
        imsize = 256
    else:
        imsize = args.imsize
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])
    from lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='test', transform=transform)
    return train_dataset, val_dataset

def custom_collate_fn(batch):
    imgs, captions, CLIP_tokens, keys, labels_tensor, BERT_input_ids, BERT_attention_mask, BERT_token_type_ids = zip(*batch)

    imgs = torch.stack(imgs)  

    #CLIP_tokens = pad_sequence([torch.tensor(token) for token in CLIP_tokens], batch_first=True, padding_value=0)
    CLIP_tokens = pad_sequence([token.clone().detach() for token in CLIP_tokens], batch_first=True, padding_value=0)

    BERT_input_ids = pad_sequence(BERT_input_ids, batch_first=True, padding_value=0)
    BERT_attention_mask = pad_sequence(BERT_attention_mask, batch_first=True, padding_value=0)
    BERT_token_type_ids = pad_sequence(BERT_token_type_ids, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels_tensor)

    return imgs, captions, CLIP_tokens, keys, labels_tensor, BERT_input_ids, BERT_attention_mask, BERT_token_type_ids

def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)
    # train dataloader
    if args.multi_gpus==True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True,
            num_workers=num_workers, sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True,
            num_workers=num_workers, shuffle='True')
    # valid dataloader
    if args.multi_gpus==True:
        valid_sampler = DistributedSampler(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True,
            num_workers=num_workers, sampler=valid_sampler)
    else:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True,
            num_workers=num_workers, shuffle='True')
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler
