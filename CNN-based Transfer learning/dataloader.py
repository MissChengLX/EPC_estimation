import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
from PIL import Image
import numpy as np
import random
from PIL import Image,ImageFilter

    
def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='vector':
        return io.loadmat(path)['pois']
    elif type=='msi':
        return io.loadmat(path)['msi']



class Mydataset_Google(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader, isMulti=None):
        with open(txt, 'r') as fh:
            file = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                file.append((words[0], int(words[1])))
                # words[0]: sentiel words[1]:Google  words[2] :labels

        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.isMulti = isMulti

    def __getitem__(self,index):

        hrs,label = self.file[index]
        if self.transform is not None:
            hrs_f = self.loader(hrs, type='msi')
            hrs_f = torch.from_numpy(hrs_f * 1.0)
            hrs_f = hrs_f.permute(1, 2, 0)  # C*H*W --> H*W*C
            hrs_f = np.array(hrs_f)
            hrs_f = Image.fromarray(hrs_f.astype('uint8'))
            hrs_f = self.transform(hrs_f)
            return hrs_f, label

    def __len__(self):
        return len(self.file)


#添加椒盐噪声
class AddSaltPepperNoise(object):

    def __init__(self, density=0,p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img

#添加Gaussian噪声
class AddGaussianNoise(object):

    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

#添加模糊
class Addblur(object):

    def __init__(self, p=0.5,blur="normal"):
        #         self.density = density
        self.p = p
        self.blur= blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            #标准模糊
            if self.blur== "normal":
                img = img.filter(ImageFilter.BLUR)
                return img
            #高斯模糊
            if self.blur== "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur)
                return img
            #均值模糊
            if self.blur== "mean":
                img = img.filter(ImageFilter.BoxBlur)
                return img

        else:
            return img


