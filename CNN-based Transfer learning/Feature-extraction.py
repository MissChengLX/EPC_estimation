#-*- coding: utf-8 -
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision
import argparse
from TF_utils import val
from torch import optim
from PIL import Image
import scipy.io as io
import numpy as np
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image
from numpy import *
from keras.applications import imagenet_utils
pca=PCA(n_components=25)

#data_txt='./Google_class_patch/data/2016_filePath.txt'
#dataPath ='./Google_class_patch/data/2017_filePath.txt'
img_to_tensor = transforms.ToTensor()
batch_size = 25
train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.RandomHorizontalFlip(p=0.3),
                                            torchvision.transforms.RandomVerticalFlip(p=0.3),
                                            torchvision.transforms.RandomRotation(60),
                                            torchvision.transforms.ToTensor(),

                                            ])
data_transform=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                               torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()
                                               #torchvision.transforms.Normalize(IMG_MEAN,IMG_STD)
                                               ])

def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='vector':
        return io.loadmat(path)['pois']
    elif type=='msi':
        return io.loadmat(path)['msi']


class Mydataset():
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        print('----------')
        with open(txt,'r') as fh:
            file = []
            X = []
            Y = []
            xlist = []
            ylist = []
            for line in fh:
                name = line.split('/')[-1].split('_')
                X = float(name[0].strip())
                Y = float(name[1].strip())
                xlist.append(X)
                ylist.append(Y)

                line=line.strip('\n')
                #line=line.rstrip()
                words=line.split()
                file.append((words[0],int(words[1]),X,Y))

        self.file=file
        self.X = X
        self.Y = Y
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader

    def __getitem__(self,index):


        lrs,label,X,Y=self.file[index]

        msi=self.loader(lrs,type='msi')
    
        if self.transform is not None:
            msi = self.transform(msi)

        return msi,label,X,Y
    def __len__(self):
        return len(self.file)

# The Google images featers extraction
class Get_feature():
    def __init__(self,datalist,Jiadao_Name,filename):
        model_val = torch.load('./Results_record/3/best_model.pth')

        model = model_val.fc[:1]  
        model = model.eval() 
        model.cuda()
        self.filename = filename
        self.model = model
        self.datalist = datalist
        self.name = Jiadao_Name
        self.extractFeatures()

    def data_generation(self,img):
        model = self.model
        img_torch = torch.unsqueeze(img, 0)
        img_torch = img_torch.float().cuda()

        outputs = model(img_torch)
        feature = outputs.data.cpu().numpy()
        feature = feature.flatten()
        return feature

    def extractFeatures(self):
        datalist = self.datalist
        name = self.name
        features =[]
        filemame = self.filename
        print(len(datalist))

        if len(datalist) < 25 and len(datalist) >=13:
            data_len = 25 - len(datalist)
            data_add = datalist[:data_len]

            for data in data_add:
                img = Image.open(data)
                img = train_transform(img)
   
                feature =self.data_generation(img)
                features.append(feature)

            for data in datalist:
                img = Image.open(data)

                img = train_transform(img)

                feature = self.data_generation(img)
                features.append(feature)

        if len(datalist) < 13:
            for data in datalist:
                img = Image.open(data)
                img = train_transform(img)
          
                feature = self.data_generation(img)
                features.append(feature)

            for i in range(5):
                if i*len(datalist) > 25:
                    break
                num =i

            for i in range(num):
                for data in datalist:
                    img = Image.open(data)
                    img = train_transform(img)
                 
                    feature =self.data_generation(img)
                    features.append(feature)
            features= features[:25]

        if len(datalist) > 25:
            for data in datalist[:25]:
                img = Image.open(data)
                img = train_transform(img)
            
                feature = self.data_generation(img)
                features.append(feature)

        features = np.array(features)

        PCA_features = pca.fit_transform(features) 

        feature_pd = pd.DataFrame(data = PCA_features)
        feature_pd.to_csv(filemame+name+'.csv', index=None)


from collections import Counter

# create new folder
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


# Extract the landscape features from high-resolution Google images at sub-district level
def Features_save():
    root_path = './Jiedao_Tif'
    # The path of sub-distirct images
    newFile = './Results_record/3/results/Jiedao_Features/'

    for dirpath, filename, filenames in os.walk(root_path): 
        for f in filename:
            mkdir(newFile+f)
            path = os.path.join(dirpath, f)
            dir_or_files = os.listdir(path)

            for fs in dir_or_files:
                jiedaoPath = os.path.join(path,fs)
                print(fs)
                # Obtain the Google images patch at sun-district level
                imgList = os.listdir(jiedaoPath)
            
                Data_list=[] 
                for img in imgList:
                    imgdir = os.path.join(jiedaoPath,img)
                    imgData = Image.open(imgdir).resize((32,32))
                  
                    img_array = np.array(imgData)
                    num = 0
                    for i in range(32):
                        for j in range(32):
                            data = img_array[i][j]
                            if np.any(data == 255):
                                num = num+1
                    if num/(32*32)*100 < 60:
                        Data_list.append(imgdir)
                        
                # Extract features from Google images at the sub-distirct level
                Get_feature(Data_list,fs.split('.')[0],newFile+f+'/')


# [2] Extract the features of each patch, and save the latitude and longitude information and feature vectors
class feature_Patch():
    def __init__(self,model,data_loader,filename):
        self.model = model
        self.filename = filename
        self.savePath = './Results_record/3/results/Google_patch_Features/' + filename.split('_')[0] + '.csv'
        X_list = []
        Y_list = []
        Features_list = []
        last_data = []
        for index, (inputs, labels, X,Y) in enumerate(data_loader):
            if index < len(data_loader)-2:
            
                # Get the extracted feature vector and the corresponding latitude and longitude coordinates
                PCA_features= self.get_features(inputs, self.model)
                PCA_features_list = np.array(PCA_features).tolist()
                for list in PCA_features_list:
                    Features_list.append(list)
                for i in X:

                    X_list.append(i.item())
                for i in Y:
                    Y_list.append(i.item())
            else:

                last_data.append(inputs)
                print(len(Y), index, inputs.shape)

                for i in X:
                    X_list.append(i.item())

                for i in Y:
                    Y_list.append(i.item())

        input_cat = torch.cat(last_data, dim=0)
        print(len(input_cat))

        PCA_features_= self.get_features(input_cat, self.model)
        for li in PCA_features_:
            Features_list.append(li)

        Features_df = pd.DataFrame(data=np.array(Features_list))

        Features_df['x'] = X_list
        Features_df['y'] = Y_list

        Features_df.to_csv(self.savePath,index=False)

    def get_features(self, img,model):
        model.eval() 
        model.cuda()
        self.img = img
        results = []

        for i in range(img.shape[0]):
            np_img = img[i,:,:]#.numpy()
            np_img = data_transform(np_img)
            img_torch = torch.unsqueeze(np_img, 0)
            img_torch = img_torch.float().cuda()
           
            outputs = model(img_torch)
            feature = outputs.data.cpu().numpy()
            feature = feature.flatten()
            results.append(feature)
        
        features = np.array(results)

        PCA_features = pca.fit_transform(features)  
        Variable_ra_list =pca.explained_variance_ratio_
        return PCA_features

# Using the trained model to extract the features of each patch from Google images
def Feature_Patch_all():
    model_val = torch.load('./best_model.pth')
    print(model_val)
    #breakpoint()
    model = model_val.fc[:1]  
    for dirpath, filename, filenames in os.walk('./Google_class_patch/data'):  # Google patch images files
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.txt':
                dataPath = dirpath + '\\' + filename
                print(filename)
                Dataset_1 = Mydataset(txt=dataPath, transform=data_transform)
                Data_loader_1 = torch.utils.data.DataLoader(Dataset_1, batch_size=batch_size, shuffle=False)
                feature_Patch(model=model, data_loader=Data_loader_1,filename=filename)

if __name__=="__main__":
    Features_save()
    Feature_Patch_all()


