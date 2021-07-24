import torch.nn as nn
import torch
from torch.autograd import Variable
from model import get_resnet
from dataloader import Mydataset_Google,Addblur
from torch.optim import lr_scheduler
import torchvision
import time
from torchsummary import summary
from Train_model import finetune
import matplotlib as plt
import argparse
import random
import os
os.chdir('I:\TFlearning_VGGF') #更改当前路径
# width = 128
# heigt = 128
width = 224
heigt =224
batch_size = 32
train_txt='./Google_class_patch/train_2013_.txt'
test_txt='./Google_class_patch/test_2013_.txt'
val_txt='./Google_class_patch/test_2013_.txt'


# Normalized data
def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion
        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape

        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def train(lr,aa,name_):

    train_tfms = torchvision.transforms.Compose([torchvision.transforms.Resize((width,heigt)),
                                                torchvision.transforms.ToTensor(),
                                                  #torchvision.transforms.Normalize(IMG_MEAN,IMG_STD)
                                                  ])

    train_ds = Mydataset_Google(txt=train_txt,transform=train_tfms, isMulti = '1')
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    data_mean, data_std = online_mean_and_sd(train_dl)
    if aa:
        train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((width,heigt)),
                                                torchvision.transforms.RandomHorizontalFlip(p=0.3),
                                                Addblur(p=1,blur="Gaussian"),

                                                #torchvision.transforms.RandomVerticalFlip(p=0.3),
                                                #torchvision.transforms.RandomCrop(size=256),
                                                #torchvision.transforms.RandomRotation(60), #90
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(data_mean,data_std)
                                                ])

        test_transform=torchvision.transforms.Compose([torchvision.transforms.Resize((width,heigt)),
                                                   torchvision.transforms.RandomHorizontalFlip(p=0.3),
                                                   Addblur(p=1, blur="Gaussian"),

                                                torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(data_mean,data_std)
                                                   ])
    else:
        train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((width, heigt)),
                                                          # torchvision.transforms.RandomHorizontalFlip(p=0.3),
                                                          # Addblur(p=1,blur="Gaussian"),

                                                          # torchvision.transforms.RandomVerticalFlip(p=0.3),
                                                          # torchvision.transforms.RandomCrop(size=256),
                                                          # torchvision.transforms.RandomRotation(60), #90
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(data_mean, data_std)
                                                          ])

        test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((width, heigt)),
                                                         # torchvision.transforms.RandomHorizontalFlip(p=0.3),
                                                         # Addblur(p=1, blur="Gaussian"),

                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(data_mean, data_std)
                                                         ])


    dataFlag = '1' # 0:sentinel 1:Google  2:muilti 3:只加载Google
    train_dataset=Mydataset_Google(txt=train_txt,transform=train_transform, isMulti = dataFlag)

    val_dataset=Mydataset_Google(txt=val_txt,transform=test_transform, isMulti = dataFlag)
    test_dataset=Mydataset_Google(txt=test_txt,transform=test_transform, isMulti = dataFlag)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    dataloader = {"train": train_loader,
                    "val": test_loader
                   }

    len_dataset = {"train": len(train_dataset),
                    "val": len(test_dataset)
                   }
    print('len trainDataset is: {}, len ValDataset is: {}'.format(len_dataset['train'],len_dataset['val']))
    model = get_resnet()
    epoch_n = 100
    model=model.cuda()
    time_open = time.time()
    model_dict = model.state_dict()
    dict_name = list(model_dict)

    for name, value in model.named_parameters():
        print(name, value.requires_grad)

    # 不同层设置不同学习率
    # optimizer = t.optim.SGD([{'params': base_params},
    #                          {'params': ignored_params.parameters(), 'lr': 0.001}], lr=0.0001)
    #optimizer=torch.optim.Adam(model.parameters(),lr=lr) #lr=0.0005 优化器

    #实现参数屏蔽 在训练时候，optimizer里面只能更新requires_grad = True的参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    #动态衰减函数
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08)
    criterion=nn.CrossEntropyLoss()  
    model_val = finetune(name_=name_,epochs_n=epoch_n,model=model, dataloaders=dataloader, optimizer=optimizer, criterion=criterion, use_lr_schedule=False)

if __name__=="__main__":
    lr=0.0001
    train(lr,True,'1')