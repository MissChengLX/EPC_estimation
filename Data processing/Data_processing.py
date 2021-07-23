#_*_ coding:utf-8 _*_
import math
import gdal
from gdalconst import *
from osgeo import gdal
import numpy as np
import random
import pandas as pd
import ogr, sys, os
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import Normalizer
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tkinter import _flatten

# New filePath
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path
        print(path + 'create file succeed')
        return True
    else:
        print(path + ' already exist!')
        return False

# Nighttime light composites data clustering using GMM method
class cluster_nighttime_data(object):
    def __init__(self):
        # Annual synthetic nighttime light data after calibrated.
        self._data1 =r'.\NPP-metadata-csv\shenzhen_汇总.csv'
    def Processing_Data(self):
        dataPath =self._data1
        data = pd.read_csv(dataPath, encoding='gbk')
        data_ = data.drop(data[data['2013_Nightlight intensity'] < 0].index)
        return data_
    def initData(self): 
        data = self.Processing_Data()
        da_list = []
        for i, col in enumerate(list(data.columns)[2:]):
            for da in data[col]:

                da_list.append(math.ceil(da))

        data_hist = pd.value_counts(da_list) 

        data_hist = pd.DataFrame(data_hist)
        data_hist['Frequency'] = data_hist[0]
        data_hist['Value'] = data_hist.index

        data_hist = data_hist.drop(0, axis=1)
        print('Getting the initData ...')
        return data_hist

    def calGMM_Everyyear(self):
        data_ = self.Processing_Data()
        random_state=87
        for i, col in enumerate(list(data_.columns)[2:]):
            data = data_[col].to_frame()
            gmm = GaussianMixture(n_components=4, covariance_type='spherical', random_state=random_state)
            cluster2 = gmm.fit(data).predict(data)
            col_name = col.split('_')[0]+'_label'
            data_[col_name] = cluster2
            print(col)
            for num in range(4):
                print(num)
                min_0 = data_[data_[col_name]==num][col].min()
                max_0 = data_[data_[col_name]==num][col].max()
                print(len(data_[data_[col_name]==num][col].tolist()))
                print(min_0,max_0)

        data_.to_csv("./NPP_classification_Results.csv")
        print('GMM classification finished!') 

# Divide the Google patch data into a trainning set and a test set
class divideTrainValidationTest(object):
    def __init__(self):
        self.train_path = './Google_class_patch/train_all.txt'
        self.test_path = './Google_class_patch/test_all.txt'
        # all the patch data for Google images
        self.data_path = './Google_class_patch/data'
        self.df_Data = pd.DataFrame(columns=['Google_path'])
        self.Google_path = []
        self.label_list = []
# Divide all data 
    def get_txtFile(self):
        for dirpath, filename, filenames in os.walk(self.data_path): 
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.txt':
                    txtPath = dirpath + '/' + filename
                    source_path = pd.read_csv(txtPath, header=None)
                    source_path['Google_path'], source_path['label'] = source_path[0].str.split(' ').str
                    self.source_path = source_path.drop([0], axis=1)
                    for i in source_path['Google_path'].tolist():
                        self.Google_path.append(i)
                    for j in source_path['label'].tolist():
                        self.label_list.append(j)
        self.df_Data['Google_path'] = self.Google_path
        self.df_Data['label'] = self.label_list
        self.divideTrainTest(self.df_Data, self.train_path, self.test_path)


    def divideTrainTest(self, source_path,train_path,test_path):
        classes_name_list = list(source_path['label'].drop_duplicates(inplace=False))
        classes_num = len(classes_name_list)
        train_data = []
        test_data = []

        for phase in ['train','test']:
            for i in range(0, classes_num):
                source_image_dir = source_path.loc[(source_path['label'] == str(i))]

                source_image_dir = source_image_dir.sample(frac=1)
                source_image_dir['path_label'] = source_image_dir['Google_path'] + ' ' + source_image_dir['label']
                source_image_dir = source_image_dir.drop(['Google_path', 'label'], axis=1)
                #70% for trainning data
                train_image_list = source_image_dir.iloc[0:int(0.7 * len(source_image_dir))]
                # test data
                test_image_list = source_image_dir.iloc[int(0.7 * len(source_image_dir)):]
                test_image_list = np.array(test_image_list).tolist()
                train_image_list = np.array(train_image_list).tolist()
                for da in train_image_list:
                    train_data.append(da)
                for da in test_image_list:
                    test_data.append(da)
            # Save the training data file address to a txt file
            if phase == 'train':
                file = open(train_path, 'w')
                for var in train_data:
                    file.writelines(var)
                    file.write('\n')
                file.close()
            else:
                file = open(test_path, 'w')
                for var in test_data:
                    file.writelines(var)
                    file.write('\n')
                file.close()


"""
    processing Google Earth images.
    Divide Google images into 32*32 patches, and the patches data save as mat format file.
"""
class Processing_Google_images(object):
    def __init__(self):
        self.dataPath = './Google Earth images in Shenzhen_L14/2013_L14'
        self.savePath ='./Google_class_patch' # file path
        self.csvFile = './X_Y_DN_pop/Shenzhen_2013.csv'
        self.matFile_path =''
        self.patch_Size = 32 
    def cal_patch_(self,tifPath, pd_NPP, Tif_Year):
        path = self.savePath
        patch_Size = self.patch_Size
        matFile_path = self.matFile_path
        mkdir(path+'\\data')
        txtFilePath = path+'\\data'+'\\'+Tif_Year+'_filePath.txt'
        in_ds = gdal.Open(tifPath)
        if in_ds != None:
            print("open tif file succeed")
        # imges bands
        im_bands = in_ds.RasterCount
        in_bands = []
        projection = in_ds.GetProjection()
        cols = in_ds.RasterXSize
        rows = in_ds.RasterYSize
        
        im_data = in_ds.ReadAsArray(0, 0, cols, rows) 

        # data types
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        print('dataType is: {}'.format(datatype))
        ori_transform = in_ds.GetGeoTransform() 
        originX = ori_transform[0]
        originY = ori_transform[3]
        pixelWidth = ori_transform[1]
        pixelHeight = ori_transform[5]
        # patch image size
        block_xsize = patch_Size  
        block_ysize = patch_Size 
        NPP_xy_pd = pd_NPP
        k = 0
        txtFile = []
        x_list =[]
        y_list =[]
        # iterate over the labels of each pixel
        for index, row in NPP_xy_pd.iterrows():
            label = row['label']
            top_left_x = row['x']
            top_left_y = row['y']

            k += 1
            # calcuted the pixel offset
            offset_x = int((top_left_x - originX) / pixelWidth)  
            offset_y = int((top_left_y - originY) / pixelHeight) 
            # whether the pixel offset is out of bounds
            if offset_y < rows:
                if (offset_y + block_ysize) < rows:
                    numRows = block_ysize
                else:
                    numRows = rows - offset_y
            else:
                continue
            if offset_x < cols:
                if (offset_x + block_xsize) < cols:
                    numCols = block_xsize
                else:
                    numCols = cols - offset_x
            else:
                continue
            # create folder for patch data (mat format)
            matFile_path = path + '/' + Tif_Year + '/'
            mkdir(matFile_path)
            matFile_name = str(top_left_x)+'_'+str(top_left_y)+'_'+str(int(label))+'.mat'
            matFile_path = matFile_path+matFile_name

            # Save each mat file path and labels
            txt = matFile_path + ' ' + str(int(label))
            txtFile.append(txt)

            out_bands = []
            for in_band in in_bands:
                out_bands.append(in_band.ReadAsArray(offset_x, offset_y, numCols, numRows))
            patch_Data =np.array(out_bands)
            self.saveTomat(patch_Data, matFile_path)
            
        # Save the generated .mat file address to a txt file
        file = open(txtFilePath, 'w')
        for var in txtFile:
            file.writelines(var)
            file.write('\n')
        file.close()
        print('save txtFile= {} success!'.format(txtFilePath))



"""
    tifPath：The filePath of worldPop GeoTIFF format file.
    svdir: The nigttime light data corresponding to worldPop, csv format.
"""
def patch_worldPop(tifPath, csvdir):
    in_ds = gdal.Open(tifPath)
    if in_ds != None:
        print("open tif file succeed")
    im_bands = in_ds.RasterCount

    in_band1 = in_ds.GetRasterBand(1)
    cols = in_ds.RasterXSize
    rows = in_ds.RasterYSize
    nodataValue = in_ds.GetRasterBand(1).GetNoDataValue()
    ori_transform = in_ds.GetGeoTransform() 

    originX = ori_transform[0]
    originY = ori_transform[3]
    pixelWidth = ori_transform[1]
    pixelHeight = ori_transform[5]
    # the patch size
    block_xsize = 5  # row
    block_ysize = 5  # loc
    NPP_xy_pd = pd.read_csv(csvdir, encoding='gbk')
    NPP_xy_pd['flag'] = NPP_xy_pd['x'].map(str) +'-'+NPP_xy_pd['y'].map(str)

    k = 0
    #The total population of each patch
    pop_df = pd.DataFrame(columns=['x_pop','y_pop','pop'])
    #Pixels and labels of each night light data.
    x=[]
    y=[]
    pop =[]
    for index, row in NPP_xy_pd.iterrows():
        label = row['label']
        top_left_x = row['x']
        top_left_y = row['y']
        k += 1
        offset_x = int((top_left_x - originX) / pixelWidth) 
        offset_y = int((top_left_y - originY) / pixelHeight) 
        # Determine whether the line number is out of bounds.
        if offset_y <rows:
            if (offset_y + block_ysize) < rows:
                numRows = block_ysize
            else:
                numRows = rows - offset_y
        else:
            continue
        if offset_x < cols:
            if (offset_x + block_xsize) < cols:
                numCols = block_xsize
            else:
                numCols = cols - offset_x
        else:
            continue

        out_band1 = in_band1.ReadAsArray(offset_x, offset_y, numCols, numRows)
        out_band1 = np.array(out_band1).flatten()
        new_out_band = np.delete(out_band1, np.where(out_band1 < 1))
        pop.append(new_out_band.sum())
        x.append(top_left_x)
        y.append(top_left_y)
    pop_df['x_pop'] = x
    pop_df['y_pop'] = y
    pop_df['pop'] = pop  # Unit: person
    pop_df['flag'] = pop_df['x_pop'].map(str) +'-'+pop_df['y_pop'].map(str)
    print(pop_df)
    mergeDF= pd.merge(NPP_xy_pd,pop_df,on='flag',how='left')
    mergeDF=mergeDF.drop(['x_pop','y_pop','flag'],axis=1)
    mergeDF.to_csv('./NPP_classification_results/pop.csv',index=False)
    print("End!")

# save the predicted EPC data as the corresponding GeoTIFF format.
def EPC_to_Tif():
    # the predicted EPC data in csv format
    osPath = './data/EPC_results_normal/'
    for dirpath, filename, filenames in os.walk(osPath): 
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.csv':
                path = dirpath + '\\' + filename
                print(filename,savePath+filename.split('.')[0]+'.tif')
                Data = pd.read_csv(path,encoding='gbk')
                Data = Data[['x','y','epc_normal','epc']]
                Data = Data.rename(columns={'epc':'epc_pre','epc_normal':'epc'})
                GRID().write_img(filename=savePath+filename.split('.')[0]+'.tif', data=Data)

# The class for writing raster data
class GRID:
    def __init__(self):
        self.filename = './NTL_TIF/Shenzhen_2013_cor.tif'
    def write_img(self,filename,data):
        im_proj, im_geotrans, im_data,nodataValue = self.read_img(data)
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
            
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape

        #create GeoTIFF format
        driver = gdal.GetDriverByName("GTiff")          
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)           
        dataset.SetProjection(im_proj)                 

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  
            print(nodataValue,type(nodataValue))
            dataset.GetRasterBand(1).SetNoDataValue(nodataValue)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
  

if __name__ == '__main__':
    
    tifPath = ''
    csvdir = ''
    # processing the worldPop dataset for extracting population density features.  
    #patch_worldPop(tifPath, csvdir)
    # save the predicted EPC data as the corresponding GeoTIFF format.
    #EPC_to_Tif()
    
    # Nighttime light data clustering
    #cluster_nighttime_data().calGMM_Everyyear()
    #divideTrainValidationTest().get_txtFile()

