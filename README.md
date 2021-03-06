# EPC_estimation
Script and data from "An assessment of electric power consumption using a random forest model with multi-source data" by Luxiao Cheng, Dong Liang, Ruyi Feng, Jining Yan, and Lizhe Wang.

## Content
This repository contains:

  1. The data processing code and related functions in the "Data processing" folder. These files also contain nessary steps in order to reproduce results.
  
    The "Data_processing.py" contains the class and fuctions as follows:
      
    patch_worldPop(tifPath, csvdir) # processing the worldPop dataset for extracting population density features.  
    
    EPC_to_Tif() # save the predicted EPC data as the corresponding GeoTIFF format.
    
    cluster_nighttime_data().calGMM_Everyyear() # Nighttime light data clustering
     
    divideTrainValidationTest().get_txtFile()
    
  2. The "NTL_Preprocessing.py" : we conducted the raw nighttime light monthly composites data caliration and synthesized annual data.

    The preprocessing of the nihttime light composites data contains three steps:
    
    Step1: Averaging the monthly nighttime light data to get the annual composites data. 
    function: avg_NPP()
    
    Step2: the negtive outliters processing
    function: binary_cal()
    
    Step3: Outlier handling for nighttime light composites data
    function: Del_maxValue()
    
  3. The "CNN-based Transfer learning" contains code files for finetruning the ResNet-50 m model to extracte the landscape features from the Google Earth images.
  
    "main.py" is the main file.
    
    "Feature-extraction.py" : This fuction is to extract landscape features based on the trained CNN-based model from Google Earth images at patch and the sub-district level. It running after "main.py".
     
     The 'resnet50_places365.pth.tar' can be download from https://github.com/kywch/Vis_places365 or can be find in the Release. 
 
 The Release contains the pre-trained model (resnet50_places365.pth.tar) and the feature extraction model (best_model.pth) after fine-turning.
 
  4. The "RFR model" contians code files for estimating EPC (electric power consumption) using a RFR model.
  
    The related code fuctions are in the "EPC_estimation.py" file.
    
    train_model() # Training a RFR model
    
    EPC_estimation() # EPC estimation based on the trained RFR model
    
    normalization(year) # Normalized the EPC estimation results from 2013 to 2019
 
 ## The operating environment
 
 The version of the required library
 
 python-3.7
 
 scikit-learn-0.24.1
 
 torch-1.7.1
 
 gdal-2.2.4
 
 ## Resources
 
 Original dataset can be found here:
 
 Journal article can be found here: 
