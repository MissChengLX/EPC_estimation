# coding=utf-8
import math
from tkinter import _flatten
from gdalconst import *
from osgeo import gdal
import numpy as np
import pandas as pd
import ogr, sys, os


def Read(RasterFile):  
    ds = gdal.Open(RasterFile, GA_ReadOnly)
    if ds is None:
        print('Cannot open ', RasterFile)
        sys.exit(1)
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    cols = ds.RasterXSize 
    rows = ds.RasterYSize 
    band = ds.GetRasterBand(2)

    im_data = band.ReadAsArray(0, 0, cols, rows)
    im_data = band.ReadAsArray()

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte 
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    noDataValue = band.GetNoDataValue()
    projection = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    im_bands = ds.RasterCount

    noDataValue = ds.GetRasterBand(1).GetNoDataValue()
    print("noDataValue = ({})".format(noDataValue))
    data = []
    if im_bands == 1:
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray(0, 0, cols, rows)

    else:
        for j in range(im_bands):
            band = ds.GetRasterBand(j + 1)

            imdatas = band.ReadAsArray(0, 0, cols, rows)
            data.append(imdatas)

    return data


# Read the raster data and return the Longitude and latitude coordinate information
def Readxy(RasterFile):  
    ds = gdal.Open(RasterFile, GA_ReadOnly)

    if ds is None:
        print('Cannot open ', RasterFile)
        sys.exit(1)
    cols = ds.RasterXSize 
    rows = ds.RasterYSize  
    im_bands = ds.RasterCount
    band = ds.GetRasterBand(1)

    im_data = band.ReadAsArray(0, 0, cols, rows)

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
        print("type=", datatype)
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    print(datatype, gdal)
    noDataValue = band.GetNoDataValue()
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()
 
    arrSlope = []  # Longitude and latitude coordinate information:X,Y
    for i in range(rows):  
        row = []
        for j in range(cols):  
            px = geotransform[0] + i * geotransform[1] + j * geotransform[2]
            py = geotransform[3] + i * geotransform[4] + j * geotransform[5]
            col = [px, py]
            row.append(col)
        arrSlope.append(row)
    return arrSlope,im_data


def WriteGTiffFile(filename, bands, nRows, nCols, data, geotrans, proj, noDataValue, gdalType):
    gdal.AllRegister()
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    ds = driver.Create(filename, nCols, nRows, 1, gdalType)
    ds.SetGeoTransform(geotrans)
    ds.SetProjection(proj)
    if noDataValue is None:
        noDataValue = 9999
    if bands == 1:
        ds.GetRasterBand(1).WriteArray(data)  # 写入数组数据
    else:
        for i in range(bands):
            ds.GetRasterBand(i + 1).WriteArray(data[i])
    del ds


def mk_trend(x):
    s = 0
    length = len(x)
    for m in range(0, length - 1):
        for n in range(m + 1, length):
            if x[n] > x[m]:
                s = s + 1
            elif x[n] == x[m]:
                s = s + 0
            else:
                s = s - 1
    return s


'''
    The preprocessing of the nihttime light composites data contains three steps:
    Step1: Averaging the monthly nighttime light data to get the annual composites data.
    Step2: the negtive outliters processing
    Step3: Outlier handling for nighttime light composites data
'''
# Step1: Averaging the monthly nighttime light data to get the annual composites data.
def avg_NPP():
    NPP = "./NPP_Shenzhen"
    NPP_year = "I:/NPP_VIIRS"
    nRows, nCols = 11337, 14781
    noDataValue, im_bands, = 0, 0
    geotrans, proj, gdalType = '', '', ''
    for i in range(2013, 2020, 1):
        npp_data = np.zeros((nRows, nCols, 9))
        AVG_data_ = np.zeros((nRows, nCols))
        num = 0
        print(num)
        for dirpath, filename, filenames in os.walk(NPP): 
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.tif':
                    cal_date = filename.split('_')[3][:4]
                    if str(i) == cal_date:
                        data = Read(dirpath + '\\' + filename)
                        data = np.array(data)
                        nRows, nCols, geotrans, proj, noDataValue, im_bands, gdalType = Readxy(dirpath + '\\' + filename)
                        print(num)
                        npp_data[:, :, num] = data
                        num = num + 1
        for r in range(nRows):
            row_list = []
            for c in range(nCols):
                avg_data = np.mean(npp_data[r, c, :]) 
                print(avg_data)
                row_list.append(avg_data)
                print(len(row_list))
            data_ = np.array(row_list)
            AVG_data_[r, :] = data_
        print(AVG_data_)
        print(max(map(max, AVG_data_)))
        saveFile = NPP_year + '\\' + str(i) + '.tif'
        WriteGTiffFile(saveFile, im_bands, nRows, nCols, AVG_data_, geotrans, proj, noDataValue, gdalType)


'''
Step2: the negtive outliters processing
binaryFile: the path of the binary data
RasterFile: the path of the raster data, which needed to be processed 
outFile: the data path after processing
'''
def binary_cal(binaryFile, RasterFile, outFile):

    ds = gdal.Open(RasterFile, GA_ReadOnly)
    if ds is None:
        print('Cannot open ', binaryFile)
        sys.exit(1)
    cols = ds.RasterXSize 
    rows = ds.RasterYSize 
    im_bands = ds.RasterCount
    band = ds.GetRasterBand(1)

    im_data = band.ReadAsArray(0, 0, cols, rows)
    # rasterfile
    ds_ = gdal.Open(RasterFile, GA_ReadOnly)
    if ds_ is None:
        print('Cannot open ', binaryFile)
        sys.exit(1)
    cols_ = ds_.RasterXSize 
    rows_ = ds_.RasterYSize  
    im_bands = ds_.RasterCount
    band_ = ds_.GetRasterBand(1)

    im_data_ = band_.ReadAsArray(0, 0, cols_, rows_)

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    noDataValue = band.GetNoDataValue()
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    
    # The negtive pixels are marked as 0, otherwise, pixels are marked as 1,
    for r in range(rows):
        for c in range(cols):
            if im_data[r][c] > 0:
                im_data[r][c] = 1
            else:
                im_data[r][c] = 0

    for r in range(rows):
        for c in range(cols):
            im_data_[r][c] = im_data[r][c] * im_data_[r][c]

    # Create the GeoTTIF file to save the processed data
    driver = gdal.GetDriverByName("GTiff")  
    dataset = driver.Create(outFile, cols, rows, im_bands, datatype)
    dataset.SetGeoTransform(geotransform)  
    dataset.SetProjection(projection) 
    dataset.GetRasterBand(1).WriteArray(im_data_) 
    del ds
    del ds_


'''
Step3: Outlier handling for nighttime light composites data
Del_maxValue()
delete_max_value()
'''
def Del_maxValue():
    df_matrix = pd.read_csv('./Max_value_China.csv')
    # get the max value in Beijing, Shanghai, Guangzhou.
    BSG = df_matrix[(df_matrix['OID'] == 2) | (df_matrix['OID'] == 5) | (df_matrix['OID'] == 24)]
    
    # The path of nighttime light composites data to be processed
    path = './Shenzhen/'
    for dirpath, filename, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.tif':
                cal_path = dirpath + '\\' + filename 
                fn = filename.split('.')[0].split('_')[1]
                print(fn)
                cal_date = int(fn)
                out ='/Del_maxValue_' + filename
                # The max value 
                maxVa = max(BSG[BSG['DBF_NAME'] == cal_date]['MAX'])
                infile = dirpath + '\\' + filename
                delete_max_value(infile, out, maxVa)            
def delete_max_value(infile, outfile, maxvalue):
    neighbor = [[-1, -1, -1, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 1, -1, 0, 1]]  # eight-neighbors
    old, rows, cols, geotrans, proj, noDataValue, im_bands, datatype = Readxy(infile)  
    r, w = np.where(old > maxvalue) 
    for i in range(len(r)):
        sum = 0
        n = 0
        for j in range(8):
            if r[i] + neighbor[0][j] < rows and (old[r[i] + neighbor[0][j]][w[i] + neighbor[1][j]] < maxvalue) and \
                    old[r[i] + neighbor[0][j]][w[i] + neighbor[1][j]] > -1000:  
                sum += old[r[i] + neighbor[0][j]][w[i] + neighbor[1][j]]
                n += 1
        old[r[i]][w[i]] = sum / n
    WriteGTiffFile(outfile, im_bands, rows, cols, old, geotrans, proj, noDataValue, datatype)

if __name__ == '__main__':
    avg_NPP()








