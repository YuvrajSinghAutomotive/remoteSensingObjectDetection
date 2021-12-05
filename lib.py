import numpy as np
import pandas as pd
import os
import cv2
from osgeo import gdal    ## used to read images in memory
import matplotlib.pyplot as plt

def dataExtract(foldername):
    data = {'filename': [],
            'MSI': [],
            'AGL': [],
            'CLS': []}
    for filename in os.listdir(foldername + '/'):
        if filename.endswith('AGL.tif'): 
            data['AGL'].append( gdal.Open(foldername + '/' + filename , gdal.GA_ReadOnly).ReadAsArray() ) 
            data['filename'].append( filename[:-8] )
        elif filename.endswith('MSI.tif'): data['MSI'].append( gdal.Open(foldername + '/' + filename , gdal.GA_ReadOnly).ReadAsArray() ) 
        elif filename.endswith('CLS.tif'): data['CLS'].append( gdal.Open(foldername + '/' + filename , gdal.GA_ReadOnly).ReadAsArray() ) 
    return pd.DataFrame(data)


def flatten(list_of_lists):
    # flatten list of lits recursively
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def dataStack(rawData):
    # Stack features for individual pixels in a long matrix: convert images to X,Y data matrices
    predictors = np.empty(shape=(0,9))
    targets = np.empty(shape=(0,1), dtype=int)

    for imgIdx in range(len(rawData)):

        # Values for each pixel in multispectral images: 8 values for every pixel
        matMSI = rawData['MSI'].iloc[imgIdx]
        matMSI = matMSI.transpose(1,2,0).reshape(-1,matMSI.shape[0])

        # Values for each pixel in height (AGL) images: 1 value for every pixel
        matAGL = rawData['AGL'].iloc[imgIdx]
        matAGL = matAGL.transpose(0,1).reshape(-1,1)

        # Values for each pixel in classes images: 1 int value for every pixel
        matCLS = rawData['CLS'].iloc[imgIdx]
        matCLS = matCLS.transpose(0,1).reshape(-1,1)

        predictors = np.concatenate( ( predictors, np.concatenate((matMSI,matAGL) , axis=1) ) , axis=0 )
        targets = np.concatenate( ( targets,matCLS ) , axis=0 )
    
    dataStack = np.concatenate((predictors,targets),axis=1)
    # targets = targets.reshape(-1)
    return dataStack

def stochasticPCAFromDataStackScratch(data,numComponents=3,numData=100000):
    randomPoints = np.random.permutation(len(data))         ## find principal components using a few random points
    dataCov = np.cov( np.array(data.iloc[randomPoints[0:numData]]).T )
    dataEigVals, dataEigVecs = np.linalg.eig(dataCov)
    PCs = np.array(data)@dataEigVecs
    return PCs[:,0:numComponents]

def stochasticPCAFromDataStack(data,numComponents=3,numData=100000):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=numComponents)
    randomPoints = np.random.permutation(len(data))         ## find principal components using a few random points
    pca.fit(data.iloc[randomPoints[0:numData]])
    return pca.transform(data)

def stochasticICAFromDataStack(data,numComponents=3,numData=100000):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=numComponents,whiten=True)
    randomPoints = np.random.permutation(len(data))         ## find principal components using a few random points
    ica.fit(data.iloc[randomPoints[0:numData]])
    return ica.transform(data)

class indicesFromDataStack:
    def __init__(self,data):
        # colNames = ['coastal','blue','green','yellow','red','red edge','near IR1', 'near IR2','AGL','CLS']
        self.data = data
    
    def ndvi(self):
        # NDVI: Normalized Difference Vegetation Index
        self.data['ndvi_NIR1'] = np.divide( ( self.data['near IR1'] - self.data['red'] ) , ( self.data['near IR1'] + self.data['red'] ) )
        self.data['ndvi_NIR2'] = np.divide( ( self.data['near IR2'] - self.data['red'] ) , ( self.data['near IR2'] + self.data['red'] ) )
        print('\tComputed NDVI: Normalized Difference Vegetation Index')
        return self.data.fillna(1)
    
    def ndwi(self):
        # NDWI: Normalized Difference Water Index
        self.data['ndwi_NIR1'] = np.divide( ( self.data['green'] - self.data['near IR1'] ) , ( self.data['green'] + self.data['near IR1'] ) )
        self.data['ndwi_NIR2'] = np.divide( ( self.data['green'] - self.data['near IR2'] ) , ( self.data['green'] + self.data['near IR2'] ) )
        print('\tComputed NDWI: Normalized Difference Water Index')
        return self.data.fillna(1)
    
    def ari(self):
        # ARI: Anthocyanin Reflectance Index
        self.data['ari'] = np.divide(np.ones(len(self.data)) , self.data['green'] ) - np.divide(np.ones(len(self.data)) , self.data['red edge'] )
        print('\tComputed ARI: Anthocyanin Reflectance Index')
        return self.data.fillna(0)
    
    def arvi(self,gamma=1):
        # ARVI: Atmospherically Resistant Vegetation Index
        # (N - (R - gamma * (R - B))) / (N + (R - gamma * (R - B)))
        self.data['arvi_NIR1'] = np.divide( self.data['near IR1'] - (self.data['red edge'] - gamma*(self.data['red edge'] - self.data['blue'])) , 
                                            self.data['near IR1'] + (self.data['red edge'] - gamma*(self.data['red edge'] - self.data['blue'])) )
        self.data['arvi_NIR2'] = np.divide( self.data['near IR2'] - (self.data['red edge'] - gamma*(self.data['red edge'] - self.data['blue'])) , 
                                            self.data['near IR2'] + (self.data['red edge'] - gamma*(self.data['red edge'] - self.data['blue'])) )
        print('\tComputed ARVI: Atmospherically Resistant Vegetation Index')
        return self.data.fillna(1)
    
    def bai(self):
        # BAI: Burned Area Index
        # 1.0 / ((0.1 - R) ** 2.0 + (0.06 - N) ** 2.0)
        self.data['bai_NIR1'] = np.divide( np.ones(len(self.data)) ,
                                          (0.1*np.ones(len(self.data)) - self.data['red edge'])**2 + (0.06*np.ones(len(self.data)) - self.data['near IR1'])**2  )
        self.data['bai_NIR2'] = np.divide( np.ones(len(self.data)) ,
                                          (0.1*np.ones(len(self.data)) - self.data['red edge'])**2 + (0.06*np.ones(len(self.data)) - self.data['near IR2'])**2  )
        print('\tComputed BAI: Burned Area Index')
        return self.data.fillna(1)
    
    def bndvi(self):
        # BNDVI: Blue Normalized Difference Vegetation Index
        # (N - B)/(N + B)
        self.data['bndvi_NIR1'] = np.divide( ( self.data['near IR1'] - self.data['blue'] ) , ( self.data['near IR1'] + self.data['blue'] ) )
        self.data['bndvi_NIR2'] = np.divide( ( self.data['near IR2'] - self.data['blue'] ) , ( self.data['near IR2'] + self.data['blue'] ) )
        print('\tComputed BNDVI: Blue Normalized Difference Vegetation Index')
        return self.data.fillna(1)
    
    def cig(self):
        # CIG: Chlorophyll Index Green
        # (N / G) - 1.0
        self.data['cig_NIR1'] = np.divide( self.data['near IR1'] , self.data['green'] ) - np.ones(len(self.data))
        self.data['cig_NIR2'] = np.divide( self.data['near IR2'] , self.data['green'] ) - np.ones(len(self.data))
        print('\tComputed CIG: Chlorophyll Index Green')
        return self.data.fillna(0)        
    
    def cire(self):
        # CIRE: Chlorophyll Index Red Edge
        # (N / RE1) - 1
        self.data['cire_NIR2'] = np.divide( self.data['near IR2'] , self.data['red edge'] ) - np.ones(len(self.data))
        self.data['cire_NIR1'] = np.divide( self.data['near IR1'] , self.data['red edge'] ) - np.ones(len(self.data))
        print('\tComputed CIRE: Chlorophyll Index Red Edge')
        return self.data.fillna(0)  
    
    def cvi(self):
        # CVI: Chlorophyll Vegetation Index
        # (N * R) / (G ** 2.0)
        self.data['cvi_NIR1'] = np.divide( np.multiply( self.data['near IR1'] , self.data['red'] ) , (self.data['green'])**2 )
        self.data['cvi_NIR2'] = np.divide( np.multiply( self.data['near IR2'] , self.data['red'] ) , (self.data['green'])**2 )
        print('\tComputed CVI: Chlorophyll Vegetation Index')
        return self.data.fillna(1)  
    
    def gbndvi(self):
        # GBNDVI: Green-Blue Normalized Difference Vegetation Index
        # (N - (G + B))/(N + (G + B))
        self.data['gbndvi_NIR1'] = np.divide( self.data['near IR1'] - ( self.data['green'] + self.data['blue'] ) , self.data['near IR1'] + ( self.data['green'] + self.data['blue'] ) )
        self.data['gbndvi_NIR2'] = np.divide( self.data['near IR2'] - ( self.data['green'] + self.data['blue'] ) , self.data['near IR2'] + ( self.data['green'] + self.data['blue'] ) )
        print('\tComputed GBNDVI: Green-Blue Normalized Difference Vegetation Index')
        return self.data.fillna(1)
    
    def gndvi(self):
        # GNDVI: Green Normalized Difference Vegetation Index
        # (N - G)/(N + G)
        self.data['gndvi_NIR1'] = np.divide( ( self.data['near IR1'] - self.data['green'] ) , ( self.data['near IR1'] + self.data['green'] ) )
        self.data['gndvi_NIR2'] = np.divide( ( self.data['near IR2'] - self.data['green'] ) , ( self.data['near IR2'] + self.data['green'] ) )
        print('\tComputed GNDVI: Green Normalized Difference Vegetation Index')
        return self.data.fillna(1)
    
    def yndvi(self):
        # YNDVI: Yellow Normalized Difference Vegetation Index
        # (N - Y)/(N + Y)
        self.data['yndvi_NIR1'] = np.divide( ( self.data['near IR1'] - self.data['yellow'] ) , ( self.data['near IR1'] + self.data['yellow'] ) )
        self.data['yndvi_NIR2'] = np.divide( ( self.data['near IR2'] - self.data['yellow'] ) , ( self.data['near IR2'] + self.data['yellow'] ) )
        print('\tComputed YNDVI: Yellow Normalized Difference Vegetation Index')
        return self.data.fillna(1)

    def grndvi(self):
        # GRNDVI: Green-Red Normalized Difference Vegetation Index
        # (N - (G + R))/(N + (G + R))
        self.data['grndvi_NIR1'] = np.divide( self.data['near IR1'] - ( self.data['green'] + self.data['red'] ) , self.data['near IR1'] + ( self.data['green'] + self.data['red'] ) )
        self.data['grndvi_NIR2'] = np.divide( self.data['near IR2'] - ( self.data['green'] + self.data['red'] ) , self.data['near IR2'] + ( self.data['green'] + self.data['red'] ) )
        print('\tComputed GRNDVI: Green-Red Normalized Difference Vegetation Index')
        return self.data.fillna(1)
    
    def mcari(self):
        # MCARI: Modified Chlorophyll Absorption in Reflectance Index
        # ((RE1 - R) - 0.2 * (RE1 - G)) * (RE1 / R)
        self.data['mcari'] = np.multiply( ( ( self.data['red edge'] - self.data['red'] ) - 0.2*( self.data['red edge'] - self.data['green'] ) ) , 
                                          np.divide( self.data['red edge'] , self.data['red'] ) )
        print('\tComputed MCARI: Modified Chlorophyll Absorption in Reflectance Index')
        return self.data.fillna(0)
    
    def mcari_osavi_ratio(self):
        # MCARI/OSAVI: MCARI/OSAVI Ratio
        # (((RE1 - R) - 0.2 * (RE1 - G)) * (RE1 / R)) / (1.16 * (N - R) / (N + R + 0.16))
        num = np.multiply( ( ( self.data['red edge'] - self.data['red'] ) - 0.2*( self.data['red edge'] - self.data['green'] ) ) , 
                           np.divide( self.data['red edge'] , self.data['red'] ) )
        den1 = np.divide( 1.16*(self.data['near IR1'] - self.data['red']) , self.data['near IR1'] + self.data['red'] + np.ones(len(self.data)) )
        den2 = np.divide( 1.16*(self.data['near IR2'] - self.data['red']) , self.data['near IR2'] + self.data['red'] + np.ones(len(self.data)) )
        self.data['mcari_osavi_ratio_NIR1'] = np.divide(num,den1)
        self.data['mcari_osavi_ratio_NIR2'] = np.divide(num,den2)
        print('\tComputed MCARI/OSAVI: MCARI/OSAVI Ratio')
        return self.data.fillna(1)

    def msavi(self):
        # MSAVI: Modified Soil-Adjusted Vegetation Index
        # 0.5 * (2.0 * N + 1 - (((2 * N + 1) ** 2) - 8 * (N - R)) ** 0.5)
        self.data['msavi_NIR1'] = 0.5 * (2.0 * self.data['near IR1'] + np.ones(len(self.data)) - (((2 * self.data['near IR1'] + np.ones(len(self.data))) ** 2) - 8 * (self.data['near IR1'] - self.data['red'])) ** 0.5)
        self.data['msavi_NIR2'] = 0.5 * (2.0 * self.data['near IR2'] + np.ones(len(self.data)) - (((2 * self.data['near IR2'] + np.ones(len(self.data))) ** 2) - 8 * (self.data['near IR2'] - self.data['red'])) ** 0.5)
        print('\tComputed MSAVI: Modified Soil-Adjusted Vegetation Index')
        return self.data.fillna(0)
    
    def nddi(self):
        # NDDI: Normalized Difference Drought Index
        # (((N - R)/(N + R)) - ((G - N)/(G + N)))/(((N - R)/(N + R)) + ((G - N)/(G + N)))
        # (term1 - term2)/(term1 + term2)
        term1_NIR1 = np.divide( (self.data['near IR1'] - self.data['red']) , (self.data['near IR1'] + self.data['red']) )
        term2_NIR1 = np.divide( (self.data['green'] - self.data['near IR1']) , (self.data['green'] + self.data['near IR1']) )
        term1_NIR2 = np.divide( (self.data['near IR2'] - self.data['red']) , (self.data['near IR2'] + self.data['red']) )
        term2_NIR2 = np.divide( (self.data['green'] - self.data['near IR2']) , (self.data['green'] + self.data['near IR2']) )
        self.data['nddi_NIR1'] = np.divide( term1_NIR1 - term2_NIR1 , term1_NIR1 + term2_NIR1 )
        self.data['nddi_NIR2'] = np.divide( term1_NIR2 - term2_NIR2 , term1_NIR2 + term2_NIR2 )
        print('\tComputed NDDI: Normalized Difference Drought Index')
        return self.data.fillna(1)
    
    def ndrei(self):
        # NDREI: Normalized Difference Red Edge Index
        # (N - RE1) / (N + RE1)
        self.data['ndrei_NIR1'] = np.divide( ( self.data['near IR1'] - self.data['red edge'] ) , ( self.data['near IR1'] + self.data['red edge'] ) )
        self.data['ndrei_NIR2'] = np.divide( ( self.data['near IR2'] - self.data['red edge'] ) , ( self.data['near IR2'] + self.data['red edge'] ) )
        print('\tComputed NDREI: Normalized Difference Red Edge Index')
        return self.data.fillna(1)
    
    def osavi(self):
        # OSAVI: Optimized Soil-Adjusted Vegetation Index
        # (N - R) / (N + R + 0.16)
        self.data['osavi_NIR1'] = np.divide( self.data['near IR1'] - self.data['red'] , self.data['near IR1'] + self.data['red'] + 0.16*np.ones(len(self.data)) )
        self.data['osavi_NIR2'] = np.divide( self.data['near IR2'] - self.data['red'] , self.data['near IR2'] + self.data['red'] + 0.16*np.ones(len(self.data)) )
        print('\tComputed OSAVI: Optimized Soil-Adjusted Vegetation Index')
        return self.data.fillna(1)

class indicesFromImages:
    def __init__(self,data):
        self.data = data
    def ndvi(self):
        pass

def stackToImage(dataStack,dataImages,colNames):
    for col in colNames:
        imgList = []
        imgStack = dataStack[col]
        for i in range( int(len(imgStack)/(1024*1024)) ):
            imgVector = np.array( imgStack.iloc[i*(1024*1024):(i+1)*(1024*1024)] )
            imgList.append( imgVector.transpose().reshape((1024,1024)) )
        dataImages[col] = imgList
    return dataImages

def vectorStackToImage(dataStack,colNames):
    for col in colNames:
        imgList = []
        imgStack = dataStack[col]
        for i in range( int(len(imgStack)/(1024*1024)) ):
            imgVector = np.array( imgStack.iloc[i*(1024*1024):(i+1)*(1024*1024)] )
            imgList.append( imgVector.transpose().reshape((1024,1024)) )
        dataImages = pd.DataFrame(np.zeros(i+1))
        dataImages[col] = imgList
    return dataImages

def featuresCompute(data, numPCA=3, numICA=3, 
                    indicesList = ['ndvi','ndwi','bai','bndvi','cvi','gndvi','yndvi','osavi']):
    predictors = data[data.columns[0:-1]]
    targets = data[data.columns[-1]]
    
    ## PCA and ICA on MSI images
    msiComponents = data[data.columns[0:-2]]
    PCs = pd.DataFrame( stochasticPCAFromDataStack(msiComponents, numComponents=numPCA, numData = len(predictors)) , 
                       columns=['PC{}'.format(i) for i in range(0,numPCA)] )
    print('Computed Principal Components')
    ICs = pd.DataFrame( stochasticICAFromDataStack(msiComponents,numComponents=numICA, numData=len(predictors)) , 
                       columns=['IC{}'.format(i) for i in range(0,numICA)])
    print('Computed Independent Components')
    predictors = pd.concat((predictors,PCs,ICs),axis=1)
    
    ## Compute Indices
    for method in indicesList:
        exec('indicesFromDataStack({}).{}()'.format('predictors',method))
    print('Computed Indices')
    
    return pd.concat((predictors,targets),axis=1)

'''
Performance Metrics
'''
def confusionMatrix(classificationTest,Ytest):
    # classificationTest: numpy array or list
    # Ytest: pandas Series or numpy array
    # Confusion Matrix: predictedLabel (rows) and actualLabel (columns)
    cMatrix = np.zeros(( len(Ytest.unique()) , len(Ytest.unique()) ))
    for idxPredictedLabel,predictedLabel in enumerate(Ytest.unique()):
        for idxActualLabel,actualLabel in enumerate(Ytest.unique()):
            predForActualLabel = classificationTest[Ytest==actualLabel]
            cMatrix[idxPredictedLabel,idxActualLabel] = np.sum(predForActualLabel == predictedLabel)
    return cMatrix

def metrics(M):
    # M: confusion matrix: predictedLabel (rows) and actualLabel (columns)
    overallAccuracy = np.sum(np.diag(M))/np.sum(M)
    userAccuracy = []
    producerAccuracy = []
    for i in range(M.shape[0]):
        userAccuracy.append(np.diag(M)[i]/np.sum(M[i,:]))
    for j in range(M.shape[1]):
        producerAccuracy.append(np.diag(M)[j]/np.sum(M[:,j]))
    N = np.sum(M)
    Mii = np.sum(np.diag(M))
    sumMplus = 0
    for i in range(M.shape[0]):
        Miplus = np.sum(M[i,:])
        Mplusi = np.sum(M[:,i])
        sumMplus = sumMplus + Miplus*Mplusi
    kappaCoeff = (N*Mii - sumMplus)/(N**2 - sumMplus)
    return overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff

def cMatrixPlots(cMatrix,YTest,MdlNames):
    ## DO NOT CALL THIS FUNCTION IN SCRIPT. Use it only in jupyter to plot confusion matrices
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
    # ax = axs.reshape(-1)
    cMatrixLabels = list(pd.Series(YTest).unique())
    # if len(cMatrixList)<=1:
    #     ax = [ax]
    # for i,cMatrix in enumerate(cMatrixList):
    img = ax.imshow(cMatrix,cmap='gray')
    ax.set_xticks(np.arange(len(cMatrixLabels)))
    ax.set_xticklabels(cMatrixLabels)
    ax.set_yticks(np.arange(len(cMatrixLabels)))
    ax.set_yticklabels(cMatrixLabels)
    ax.set_xlabel('Class (Actual)')
    ax.set_ylabel('Class (Predicted)')
    ax.set_title(MdlNames)
    for j in range(len(cMatrixLabels)):
        for k in range(len(cMatrixLabels)):
            ax.text(j-0.35,k,int(cMatrix[k,j]),color='blue',fontweight='semibold',fontsize=18)
    fig.colorbar(mappable=img,ax = ax, fraction=0.1)
    fig.tight_layout()
    return fig,ax
