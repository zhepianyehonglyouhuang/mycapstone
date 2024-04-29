import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import clip as clip
import torchvision
import torch.nn as nn


def get_fix_data(train_dl, test_dl, text_encoder, args):
    fixed_image_train, _, _, fixed_sent_train, fixed_word_train, fixed_key_train, fixed_labels_train, _, _, _= get_one_batch_data(train_dl, text_encoder, args) #0331 labels
    fixed_image_test, _, _, fixed_sent_test, fixed_word_test, fixed_key_test, fixed_labels_test, _, _, _= get_one_batch_data(test_dl, text_encoder, args) #0331 labels
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    ##################
    fixed_labels = torch.cat((fixed_labels_train, fixed_labels_test), dim=0) #0331更新
    #####################
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    #fixed_BERT_input_ids = torch.cat((fixed_BERT_input_ids_train, fixed_BERT_input_ids_test), dim=0)
    #fixed_BERT_attention_mask = torch.cat((fixed_BERT_attention_mask_train, fixed_BERT_attention_mask_test), dim=0)
    #fixed_BERT_token_type_ids = torch.cat((fixed_BERT_token_type_ids_train, fixed_BERT_token_type_ids_test), dim=0)
    return fixed_image, fixed_sent, fixed_word, fixed_noise ,fixed_labels# 0331更新


def get_one_batch_data(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    imgs, captions, CLIP_tokens, sent_emb, words_embs, keys, labels, BERT_input_ids, BERT_attention_mask, BERT_token_type_ids = prepare_data(data, text_encoder, args.device) #0331更新
    # 打印标签和映射，以确认标签确实存在于映射中
    print("Sample labels:", labels[:10])  # 打印前几个标签作为样本
    print("Label to index map:", dataloader.dataset.label_to_index)  # 打印映射
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys, labels, BERT_input_ids, BERT_attention_mask, BERT_token_type_ids #0331更新


def prepare_data(data, text_encoder, device):
    #imgs, captions, CLIP_tokens, keys = data
    imgs, captions, CLIP_tokens, keys, labels, BERT_input_ids, BERT_attention_mask, BERT_token_type_ids= data #0331更新
    imgs, CLIP_tokens = imgs.to(device), CLIP_tokens.to(device)
    sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys , labels, BERT_input_ids, BERT_attention_mask, BERT_token_type_ids #0331更新


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img


def get_caption(cap_path,clip_info):
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)
    sent_ix = random.randint(0, len(eff_captions))
    caption = eff_captions[sent_ix]
    tokens = clip.tokenize(caption,truncate=True)
    return eff_captions, caption, tokens[0]
###################0407更新

def get_labels_and_mapping(base_path):
    # List all subdirectories at base_path, which are considered as label names
    labels = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    labels.sort()  # Optionally sort the labels

    # Create a mapping from label names to integers
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    return labels, label_to_index

# Obtain labels and their indices using the provided method
base_path = '/content/drive/My Drive/capstone5703/GALIP/data/movie' ###这里每次换数据集需要修改###
labels, label_to_index = get_labels_and_mapping(base_path)
# print("Label list:", labels)
# print("Label mapping:", label_to_index)
from transformers import BertModel, BertTokenizer
BERT_token = BertTokenizer.from_pretrained('bert-base-chinese')
def get_BERT(eff_captions):
    sents = [str(caption) for caption in eff_captions]
    eff_captions = BERT_token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=300,
                                    return_tensors='pt',)
    BERT_input_ids = eff_captions['input_ids']
    BERT_attention_mask = eff_captions['attention_mask']
    BERT_token_type_ids = eff_captions.get('token_type_ids',None)
    return BERT_input_ids, BERT_attention_mask, BERT_token_type_ids
##################################################

################################################################
#                    Dataset
################################################################
class TextImgDataset(data.Dataset):
    def __init__(self, split, transform=None, args=None):
        self.transform = transform
        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.split=split
        
        if self.data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        self.split_dir = os.path.join(self.data_dir, split)
        self.filenames = self.load_filenames(self.data_dir, split)
        self.number_example = len(self.filenames)
        # 收集标签 0331
        # 初始化标签映射
        self.label_to_index = self.create_label_mapping()
    #######更新
    def create_label_mapping(self):
        # 收集标签的逻辑
        unique_labels = set()
        for filename in self.filenames:
            label = filename.split('/')[0]  # 根据你的 __getitem__ 方法获取标签
            unique_labels.add(label)
        # 创建从标签到索引的映射
        label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}  # 修改点：确保一致性，通过排序
        #print(label_to_index)  # 在 __init__ 方法的末尾添加这行代码来检查映射
        return label_to_index

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        #
        if self.dataset_name.lower().find('coco') != -1:
            if self.split=='train':
                img_name = '%s/images/train2014/jpg/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/images/val2014/jpg/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
        elif self.dataset_name.lower().find('cc3m') != -1:
            if self.split=='train':
                img_name = '%s/images/train/%s.jpg' % (data_dir, key)
                text_name = '%s/text/train/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/test/%s.jpg' % (data_dir, key)
                text_name = '%s/text/test/%s.txt' % (data_dir, key.split('_')[0])
        elif self.dataset_name.lower().find('cc12m') != -1:
            if self.split=='train':
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
        else:
            img_name = '%s/images/%s.jpg' % (data_dir, key)
            text_name = '%s/text/%s.txt' % (data_dir, key)
        #
        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        eff_captions, caps,tokens = get_caption(text_name,self.clip4text)
        BERT_input_ids, BERT_attention_mask, BERT_token_type_ids = get_BERT(eff_captions)

        label_name = key.split('/')[0] ####0331更新
        label_index = self.label_to_index[label_name]  # 获取标签索引
        label_tensor = torch.tensor(label_index, dtype=torch.long)  # 转换为长整型Tensor  ###4.17

        return imgs, caps, tokens, key , label_tensor, BERT_input_ids, BERT_attention_mask, BERT_token_type_ids###0331更新label###4.17

    def __len__(self):
        return len(self.filenames)
# =====================resnet=========================
def load_resnet(pretrained=True): 
    # load pretraned ResNet50
    resnet = torchvision.models.resnet50(pretrained=pretrained)

    # modify a specific residual block that includes downsampling
    block_to_modify = resnet.layer4[0] 

    # adjust the shortcut path to match this
    if block_to_modify.downsample is None:
        block_to_modify.downsample = nn.Sequential(
            nn.Conv2d(block_to_modify.conv1.in_channels, block_to_modify.conv3.out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(block_to_modify.conv3.out_channels)
        )
    else:
        block_to_modify.downsample[0].stride = (2, 2)

    # move the model to GPU and convert to half precision
    resnet = resnet.cuda().half()

    return resnet