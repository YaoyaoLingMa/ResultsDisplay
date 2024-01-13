import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import BBDataset
from torch.utils.data import DataLoader
from models.model import SAAN
from models.model_atten import SAAN_ATTEN
from models.model_vit import SAAN_VIT
import torch.optim as optim
from common import *
import argparse
from tqdm import tqdm

CUDA_INDEX = 'cuda:2'
LAMDA = 0.67
patience = 20  # 如果连续7个epoch验证集损失没有降低，就停止训练
train_dataset = BBDataset(file_dir='/home/liukai/likeyao/BAID/dataset', type='train', test=False)
val_dataset = BBDataset(file_dir='/home/liukai/likeyao/BAID/dataset', type='validation', test=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/liukai/likeyao/BAID/checkpoint/SAAN-pre39-huberloss-0.67')
    parser.add_argument('--val_freq', type=int,
                        default=2)
    parser.add_argument('--save_freq', type=int,
                        default=3)

    return parser.parse_args()




#
def validate(args, model, val_loader, epoch):
    model.eval()
    if args.device == 'cuda' and torch.cuda.is_available():
        device = CUDA_INDEX
    else:
        device = 'cpu'

    loss = nn.MSELoss()
    val_loss = 0.0
    with torch.no_grad():
        for step, val_data in enumerate(val_loader):
            image = val_data[0].to(device)
            label = val_data[1].to(device).float()

            predicted_label = model(image).squeeze()
            val_loss += loss(predicted_label, label).item()

    val_loss /= len(val_loader)
    print("Epoch: %3d Validation loss: %.8f" % (epoch, val_loss))
    return val_loss

def emd_loss(p_target: torch.Tensor, p_estimate: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()

# Huber Loss function
def HuberLoss(y_true, y_pred,delta=1.0):
    # 计算预测误差的绝对值
    error = torch.abs(y_true - y_pred)
    # 通过 torch.clamp 函数将误差限制在 [0.0, delta] 的范围内
    quadratic_part = torch.clamp(error, 0.0, delta)
    # 计算误差中超过 delta 阈值的部分
    linear_part = error - quadratic_part
    # 计算完整的 Huber Loss，由平方项和线性项组成
    loss = 0.5 * quadratic_part**2 + delta * linear_part
    # 返回损失的平均值
    return torch.mean(loss)


def train(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = CUDA_INDEX
    else:
        device = 'cpu'
    print('Running Training on '+device)
    model = SAAN(num_classes=1)
    for name, param in model.named_parameters():
        if 'GenAes' in name:
            param.requires_grad = False
    model = model.to(device)

    loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,drop_last=True)

    best_val_loss = float('inf')
    
    counter = 0

    for epoch in tqdm(range(args.epoch),desc='Epoch:'):
        model.train()
        epoch_loss = 0.0

        for step, train_data in tqdm(enumerate(train_loader),desc='Step:'):
            optimizer.zero_grad()
            image = train_data[0].to(device)
            label = train_data[1].to(device).float()

            predicted_label = model(image).squeeze()

            #计算Huber Loss
            huber_loss = HuberLoss(label,predicted_label,1.0)
            #计算mse loss
            mse_loss = loss(predicted_label, label)
            #两者按一定比例相加得到最终loss，比例通过LAMDA控制
            total_loss = LAMDA*mse_loss + (1.0-LAMDA)*huber_loss
            total_loss.backward()
            
            optimizer.step()
            
            epoch_loss += total_loss.item() 
            

            print("Epoch: %3d Step: %5d / %5d Train loss: %.8f" % (epoch, step, len(train_loader),total_loss.item()))

        adjust_learning_rate(args, optimizer, epoch)

        
        val_loss = validate(args, model, val_loader, epoch)

        # 检查是否需要早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} because validation loss did not improve.")
                save_checkpoint(args, model, epoch)
                break

        if (epoch + 1) % args.save_freq == 0 or counter == 0:
            save_checkpoint(args, model, epoch)

    print('Training Complete!!!')


if __name__ == '__main__':
    args = parse_args()
    train(args)
