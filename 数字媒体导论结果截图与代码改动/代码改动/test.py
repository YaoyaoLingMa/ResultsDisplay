import os
import random
import scipy.stats
from data import BBDataset,RedditArtDataset,RedditPhotocritiqueDataset
from torch.utils.data import DataLoader
from models.model import SAAN
from models.model_atten import SAAN_ATTEN
from models.model_vit import SAAN_VIT
import torch.nn as nn
import torch
import torch.optim as optim
from common import *
import argparse
import torch.nn.functional as F
import pandas as pd
import scipy
from tqdm import tqdm

#test_dataset = BBDataset(file_dir='/home/liukai/likeyao/BAID/dataset', type='test', test=True)
#test_dataset = RedditArtDataset()
test_dataset = RedditPhotocritiqueDataset()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/liukai/likeyao/BAID/checkpoint/SAAN-pre39-add-vit-0.3')
    parser.add_argument('--checkpoint_name', type=str,
                        default='epoch_29.pth')
    parser.add_argument('--save_dir', type=str,
                        default='result')

    return parser.parse_args()



def test(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    #df = pd.read_csv('/home/liukai/likeyao/BAID/dataset/test_set.csv')
    df = pd.read_csv('/home/liukai/likeyao/AestheticScore/DataSet/3000/image_score_512_train.csv')
    predictions = []

    model = SAAN_VIT(num_classes=1)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    with torch.no_grad():
        for step, test_data in tqdm(enumerate(test_loader)):
            image = test_data[0].to(device)

            predicted_label = model(image)
            prediction = predicted_label.squeeze().cpu().numpy()
            predictions.append(prediction * 10)

    scores = df['score'].values.tolist()
    #reddit_art_path = '/data/liukai/likeyao/images/ArtFromReddit/'
    #image_files = [f for f in os.listdir(reddit_art_path+'images') if f.endswith(('.png', '.jpg', '.jpeg'))]
    #data = []
    #for image_file in image_files:
    #    random_number = round(random.uniform(0, 10), 2)
    #    data.append([image_file, random_number])
    #DATA = pd.DataFrame(data,columns = ['image','score'])
    #scores = DATA['score'].values.tolist()
    #

    print(scipy.stats.spearmanr(scores, predictions))
    print(scipy.stats.pearsonr(scores, predictions))

    acc = 0
    for i in range(len(scores)):
        cls1 = 1 if scores[i] > 5 else 0
        cls2 = 1 if predictions[i] > 5 else 0
        if cls1 == cls2:
            acc += 1
    print(acc/len(scores))

    df.insert(loc=2, column='prediction', value=predictions)
    #DATA.insert(loc=2, column='prediction', value=predictions)\
    #

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, 'result.csv')

    df.to_csv(save_path, index=False)
    #DATA.to_csv(save_path, index=False)
    #


if __name__ == '__main__':
    args = parse_args()
    test(args)
