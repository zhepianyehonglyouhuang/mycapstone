import os, sys
from pyexpat import features
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from lib.utils import transf_to_CLIP_input, dummy_context_mgr
from lib.utils import mkdir_p, get_rank
from lib.datasets import prepare_data, load_resnet

from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
from lib.network import MultiModal, CrossModule, UnimodalDetection
#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#criterionC = nn.CrossEntropyLoss()
############   GAN   ############

def train(dataloader, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, optimizerC, scaler_G, scaler_D, BERT, args, criterionC, criterionD):#0424
    netG.to(args.device)
    netD.to(args.device)
    netC.to(args.device)
    text_encoder.to(args.device)
    image_encoder.to(args.device)
    BERT.to(args.device)
    
    num_classes = 4
    device = args.device
    batch_size = args.batch_size
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC, image_encoder = netG.train(), netD.train(), netC.train(), image_encoder.train()
    BERT = BERT.train()
    resnet = load_resnet(True)
    projection = MultiModal()
    projection.to(args.device)
    
    for epoch in range(max_epoch):
        total_discriminator_loss = 0
        total_classification_loss = 0
        num_steps = 0

        resnet = load_resnet(pretrained=True)
        resnet = resnet.to(args.device)

        loop = tqdm(total=len(dataloader), desc=f'Train Epoch [{epoch+1}/{max_epoch}]')

        for step, data in enumerate(dataloader, 0):
            real, captions, CLIP_tokens, sent_emb, words_embs, keys, labels, input_ids, attention_mask, token_type_ids  = prepare_data(data, text_encoder, args.device)
            labels = labels.to(args.device)
            input_ids = input_ids[:, 0, :]  # Select the first sequence of each batch#0420update
            attention_mask = attention_mask[:, 0, :]#0420update
            token_type_ids = token_type_ids[:, 0, :] #0420update
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            BERT_feature = BERT(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            last_hidden_states = BERT_feature['last_hidden_state']
            all_hidden_states =  BERT_feature['hidden_states']

            real.requires_grad_()
            sent_emb.requires_grad_()#名字不能改的
            words_embs.requires_grad_()
            
            optimizerD.zero_grad()
            optimizerC.zero_grad()#0419
            
            with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
                CLIP_real, real_emb = image_encoder(real)
                resnet_real = resnet(real.cuda().half())
                print("Resnet_Real shape:", resnet_real.shape)
                print("CLIP_real shape:", CLIP_real.shape)
                print("sent_emb shape:", sent_emb.shape)
                print("sent_emb:", sent_emb)
                # reshape resnet_real to have the same number of dimensions as CLIP_real
                #resnet_real = resnet_real.view(8, 1000, 1, 1, 1).expand(-1, -1, 768, 7, 7)
                # concatenate along the channel dimension
                #img_real = torch.cat((CLIP_real, resnet_real_expanded), dim=1).detach()  # concatenating along the channel dimension
                new_sent_emb, img_real, correlation, final_feature = projection(last_hidden_states, all_hidden_states, resnet_real, sent_emb, CLIP_real)
                print("new_sent_emb shape:", new_sent_emb.shape)
                real_feats = netD(img_real, labels)
                #real_feats = netD(CLIP_real, labels)
                real_out_discriminator, logits, probs, predicted_classes = netC(real_feats, new_sent_emb, labels)
                mis_sent_emb = torch.cat((new_sent_emb[1:], new_sent_emb[0:1]), dim=0).detach()
                noise = torch.randn(batch_size, z_dim, device=device)
                fake, _ = netG(noise, new_sent_emb, labels)
                CLIP_fake, fake_emb = image_encoder(fake)
                fake_feats = netD(CLIP_fake.detach(), labels)
                
                real_loss = criterionD(netC, real_feats, new_sent_emb, labels, negative=False)
                fake_loss = criterionD(netC, fake_feats, new_sent_emb, labels, negative=True)
                
                discriminator_loss = (real_loss + fake_loss) / 2
                classification_loss = criterionC(logits, labels).to(device)

                '''if args.mixed_precision:
                    scaler_D.scale(discriminator_loss + classification_loss).backward()
                    scaler_D.step(optimizerD)
                    scaler_D.update()
                else:
                    (discriminator_loss + classification_loss).backward()
                    optimizerD.step()
                '''
                #0419
                optimizerD.zero_grad()
                optimizerC.zero_grad()
                if args.mixed_precision:
                    scaler_D.scale(discriminator_loss).backward(retain_graph=True)
                    scaler_D.scale(classification_loss).backward()
                    scaler_D.step(optimizerD)
                    scaler_D.step(optimizerC)  # Update the classifier's parameters
                    scaler_D.update()
                else:
                    discriminator_loss.backward(retain_graph=True)
                    classification_loss.backward()
                    optimizerD.step()
                    optimizerC.step()  # Update the classifier's parameters
                    optimizerC.zero_grad()
                    ####
                total_discriminator_loss += discriminator_loss.item()
                total_classification_loss += classification_loss.item()
                num_steps += 1
            
            optimizerG.zero_grad()
            with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
                fake_feats = netD(CLIP_fake.detach(), labels)
                output_discriminator, _, _, _ = netC(fake_feats, new_sent_emb, labels)
                output = output_discriminator
                fake_feats = fake_feats.detach()
                output = output.detach()
                text_img_sim = torch.cosine_similarity(fake_emb, new_sent_emb).mean()
                errG = -output.mean() - args.sim_w*text_img_sim
                print("errG:", errG)
                print("scaler_G:", scaler_G)
                if args.mixed_precision:
                    scaler_G.scale(errG).backward()
                    scaler_G.step(optimizerG)
                    scaler_G.update()
                else:
                    errG.backward()
                    optimizerG.step()
                    optimizerG.zero_grad()
                    torch.cuda.empty_cache()
            
            loop.update(1)

        loop.close()
        # Calculate average losses for the epoch
        avg_discriminator_loss = total_discriminator_loss / num_steps
        avg_classification_loss = total_classification_loss / num_steps
        print(f"Epoch [{epoch+1}/{max_epoch}] Average Discriminator Loss: {avg_discriminator_loss}, Average Classification Loss: {avg_classification_loss}")





def test(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    FID, TI_sim = calculate_FID_CLIP_sim(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size)
    return FID, TI_sim


def save_model(netG, netD, netC, optG, optD, epoch, multi_gpus, step, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d_%03d.pth' % (save_path, epoch, step))


#########   MAGP   ########
def MA_GP_MP(img, sent, out, scaler):
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    #inv_scale = 1./scaler.get_scale()
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)                        
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def MA_GP_FP32(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp):
    netG.eval()
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        real, captions, CLIP_tokens, sent_emb, words_embs, keys, labels = prepare_data(data, text_encoder, device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            fake_imgs = torch.clamp(fake_imgs, -1., 1.)
            if multi_gpus==True:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            else:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            mkdir_p(batch_img_save_dir)
            vutils.save_image(fake_imgs.data, batch_img_save_name, nrow=8, value_range=(-1, 1), normalize=True)
            mkdir_p(batch_txt_save_dir)
            txt = open(batch_txt_save_name,'w')
            for cap in captions:
                txt.write(cap+'\n')
            txt.close()
            for j in range(batch_size):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                ######################################################
                # (3) Save fake images
                ######################################################      
                if multi_gpus==True:
                    single_img_name = 'batch_%04d.png'%(j)
                    single_img_save_dir  = osp.join(save_dir, 'single', str('gpu%d'%(get_rank())), 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)
                else:
                    single_img_name = 'step_%04d.png'%(step)
                    single_img_save_dir  = osp.join(save_dir, 'single', 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)   
                mkdir_p(single_img_save_dir)   
                im.save(single_img_save_name)
        if (multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('Step: %d' % (step))


'''def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    """ Calculates the FID """
    clip_cos = torch.FloatTensor([0.0]).to(device)
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    # n_gpu = dist.get_world_size()
    n_gpu = 1
    print(n_gpu)
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys, labels = prepare_data(data, text_encoder, device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb,eval=True).float()
                # norm_ip(fake_imgs, -1, 1)
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                if epoch==-1:
                    loop.set_description('Evaluating]')
                else:
                    loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    # CLIP-score
    CLIP_score_gather = list(torch.empty_like(clip_cos) for _ in range(n_gpu))
    # dist.all_gather(CLIP_score_gather, clip_cos)
    clip_score = torch.cat(CLIP_score_gather, dim=0).mean().item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value,clip_score'''

################################################### 0418 hd 替换全部的calculate_FID_CLIP_sim
def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    clip_cos = torch.FloatTensor([0.0]).to(device)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
    ])

    n_gpu = 1
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))

    loop = tqdm(total=int(dl_length * times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys, labels, _, _, _= prepare_data(data, text_encoder, device)
            
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, z_dim).to(device)
                # Unpack the tuple from netG and convert images to float
                fake_imgs, _ = netG(noise, sent_emb, labels, eval=True)  # Updated here
                fake_imgs = fake_imgs.float()  # Now correctly handling the image tensor
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze().cpu().data.numpy()
                pred_arr[start:end] = pred
            
            loop.update(1)
            if epoch == -1:
                loop.set_description('Evaluating')
            else:
                loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')

    loop.close()
    clip_score = clip_cos.mean().item() / (dl_length * times)
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value, clip_score
############################################################################ end




def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim

'''
def sample_one_batch(noise, sent, netG, multi_gpus, epoch, img_save_dir, writer):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        netG.eval()
        with torch.no_grad():
            B = noise.size(0)
            fixed_results_train = generate_samples(noise[:B//2], sent[:B//2], netG).cpu()
            torch.cuda.empty_cache()
            fixed_results_test = generate_samples(noise[B//2:], sent[B//2:], netG).cpu()
            torch.cuda.empty_cache()
            fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, value_range=(-1, 1), normalize=True)

def generate_samples(noise, caption, model):
    with torch.no_grad():
        fake = model(noise, caption, eval=True)
    return fake
'''

############################### 0418 hd 替换全部的sample_one_batch

def sample_one_batch(noise, sent, labels, model, multi_gpus, epoch, img_save_dir, writer):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        model.eval()
        with torch.no_grad():
            B = noise.size(0)
            fixed_results_train = generate_samples(noise[:B//2], sent[:B//2], labels[:B//2], model, eval_mode=True)
            torch.cuda.empty_cache()
            fixed_results_test = generate_samples(noise[B//2:], sent[B//2:], labels[B//2:], model, eval_mode=True)
            torch.cuda.empty_cache()
            fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
            
            # 确保张量在保存前为 float32 类型，且在 CPU 上
            fixed_results = fixed_results.to(dtype=torch.float32, device='cpu')
        
        img_name = 'samples_epoch_%03d.png' % (epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, value_range=(-1, 1), normalize=True)
      
        
################################# 0418 hd  替换全部的generate_samples      
def generate_samples(noise, caption, label, model, eval_mode=False):
    if eval_mode:
        model.eval()  # 设置模型为评估模式
    else:
        model.train()  # 设置模型为训练模式

    with torch.no_grad():  # 关闭梯度计算，适用于模型评估或生成样本
        fake, _ = model(noise, caption, label, eval=eval_mode)  # 解包元组，只处理图像部分
        fake = fake.cpu()  # 将图像移动到 CPU

    if eval_mode:
        model.train()  # 评估后重置为训练模式

    return fake
###################


#################0419 替换全部的predict_loss
def predict_loss(predictor, img_feature, text_feature, labels, negative): #0419
    outputs = predictor(img_feature, text_feature, labels)
    output = outputs[0] if isinstance(outputs, tuple) else outputs
    err = hinge_loss(output, negative)
    return output, err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
