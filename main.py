from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import shutil
import random
import numpy as np
import utils
import os
import random

imageWidth, imageHeight = 93, 75
amountCroppedSplitDomino = 300 #len(os.listdir('screenshots/croppedDomino')) - 10
amountCroppedSplitNoDomino = 400 #len(os.listdir('screenshots/croppedNoDomino')) - 10
testAmount = int(amountCroppedSplitDomino / 5)
trainAmount = int(amountCroppedSplitDomino / 5 * 4)


def fetchImagesFromPool(train, test):
    utils.fetchImagesFromPool(1, 1+train, 'screenshots/croppedNoDomino', 'screenshots/train/noDomino')
    utils.fetchImagesFromPool(1, 1+train, 'screenshots/croppedDomino', 'screenshots/train/domino')
    utils.fetchImagesFromPool(1+train, 1+train+test, 'screenshots/croppedNoDomino', 'screenshots/test/noDomino')
    utils.fetchImagesFromPool(1+train, 1+train+test, 'screenshots/croppedDomino', 'screenshots/test/domino')

def emptyTrainAndTestFolders():
    utils.emptyFolder('screenshots/train/noDomino')
    utils.emptyFolder('screenshots/train/domino')
    utils.emptyFolder('screenshots/test/noDomino')
    utils.emptyFolder('screenshots/test/domino')

def trainModel1(train, epochs=5, folds=5):
    print('Training model with', epochs, 'epochs...')
    imgWidth, imgHeight = imageWidth, imageHeight

    trainDataDir = 'screenshots/trainSplit'
    testDataDir = 'screenshots/testSplit'

    trainSamples = trainAmount
    testSamples = testAmount

    epochs = epochs #should be more like 50
    batchSize = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (3, imgWidth, imgHeight)
    else:
        input_shape = (imgWidth, imgHeight, 3)


    for i in range(folds):
        splitPrepare(trainAmount, testAmount)

        model = getModel(input_shape)

        trainDataGen = ImageDataGenerator(rescale=1./255)
        testDataGen = ImageDataGenerator(rescale=1./255)

        trainingSet = trainDataGen.flow_from_directory(trainDataDir, target_size=(imgWidth, imgHeight),
            batch_size=batchSize,
            class_mode='binary')

        testSet = testDataGen.flow_from_directory(
            testDataDir,
            target_size=(imgWidth, imgHeight),
            batch_size=batchSize,
            class_mode='binary')

        ''' Returns:
        A DirectoryIterator yielding tuples of(cross, y) where cross is a numpy array containing a batch of images with shape(batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
        
    
        print(len(trainingSet)) # anzahl bilder / batchSize
        print(len(trainingSet[0])) # tuple
        print(len(trainingSet[1])) # tuple
        print(len(trainingSet[2])) # tuple
    
        print(len(trainingSet[0][0])) # tuple 0, numpy array 0 (numpy array containing a batch of images with shape(batch_size, *target_size, channels)) = batch of 32 images
        print(trainingSet[0][0][0].shape)
        print(len(trainingSet[0][0][0][0])) # erstes image
    
        print(trainingSet[0][0][0][0]) # erstes image
    
        #print(trainingSet[0][0])
        #print(trainingSet[1][0])
        '''

        if train:
            print('training...')
            output = model.fit_generator(
                trainingSet,
                steps_per_epoch=trainSamples // batchSize,
                epochs=epochs,
                validation_data=testSet,
                validation_steps=testSamples // batchSize,
                verbose=2)

            model.save_weights('firstTry.h5')
        else:
            model.load_weights('firstTry.h5')


        validateImages(model, trainingSet)


def getModel(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def validateImages(model, trainingSet):
    print(trainingSet.class_indices)
    for file in os.listdir('screenshots/validationFolder'):
        if not file.startswith('.'):
            filePlusDir = os.path.join('screenshots/validationFolder', file)
            testImage = image.load_img(filePlusDir, target_size=(imageWidth, imageHeight))
            testImage = image.img_to_array(testImage)
            testImage = np.expand_dims(testImage, axis=0)
            result = model.predict(testImage)
            if result[0][0] == 0:
                prediction = 'domino'
            else:
                prediction = 'no domino'

            print(file + ': ' + prediction)
            print(result)


def prepare():
    emptyTrainAndTestFolders()
    utils.emptyFolder('screenshots/resizedDomino')
    utils.emptyFolder('screenshots/resizedNoDomino')
    utils.emptyFolder('screenshots/originalDomino')
    utils.emptyFolder('screenshots/originalNoDomino')
    utils.emptyFolder('screenshots/validationFolder')
    utils.emptyFolder('screenshots/croppedDomino')
    utils.emptyFolder('screenshots/croppedNoDomino')
    utils.getCoordinatesFile()
    utils.fetchImagesFromSourceToOriginalFolders()
    utils.splitAndCropImagesDominoScreenshots('screenshots/originalDomino')
    utils.splitAndCropImagesNoDominoScreenshots('screenshots/originalNoDomino')


    # copies the test images into the folders
    for file in os.listdir('screenshots/croppedDomino')[:testAmount]:
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/test/domino', file)
            shutil.copy(os.path.join('screenshots/croppedDomino', file), filePathToSave)
    for file in os.listdir('screenshots/croppedNoDomino')[:testAmount]:
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/test/noDomino', file)
            shutil.copy(os.path.join('screenshots/croppedNoDomino', file), filePathToSave)
    # copies the train images into the folders
    for file in os.listdir('screenshots/croppedDomino')[testAmount:amountCroppedSplitDomino]:
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/train/Domino', file)
            shutil.copy(os.path.join('screenshots/croppedDomino', file), filePathToSave)
    for file in os.listdir('screenshots/croppedNoDomino')[testAmount:amountCroppedSplitNoDomino]:
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/train/noDomino', file)
            shutil.copy(os.path.join('screenshots/croppedNoDomino', file), filePathToSave)
    # copies the last 10 pictures of each class into the validation folder
    #for file in random.sample(os.listdir('screenshots/croppedDomino'), 10):
    for file in random.sample(os.listdir('screenshots/croppedDomino')[:30], 10):
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/validationFolder/' , 'domino' + file)
            shutil.copy(os.path.join('screenshots/croppedDomino', file), filePathToSave)
    #for file in random.sample(os.listdir('screenshots/croppedNoDomino'), 10):
    for file in random.sample(os.listdir('screenshots/croppedNoDomino')[:30], 10):
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/validationFolder/', 'noDomino' + file)
            shutil.copy(os.path.join('screenshots/croppedNoDomino', file), filePathToSave)
    #utils.resizeImagesInFolder((imageWidth, imageHeight), 'screenshots/validationFolder')

    return (testAmount, trainAmount)




def splitPrepare(trainSampleSize, testSampleSize):
    utils.emptyFolder('screenshots/testSplit/domino')
    utils.emptyFolder('screenshots/testSplit/noDomino')
    utils.emptyFolder('screenshots/trainSplit/domino')
    utils.emptyFolder('screenshots/trainSplit/noDomino')

    for file in random.sample(os.listdir('screenshots/croppedDomino'), testSampleSize):
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/testSplit/domino', file)
            shutil.copy(os.path.join('screenshots/croppedDomino', file), filePathToSave)
    for file in random.sample(os.listdir('screenshots/croppedNoDomino'), testSampleSize):
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/testSplit/noDomino', file)
            shutil.copy(os.path.join('screenshots/croppedNoDomino', file), filePathToSave)

    for file in random.sample(os.listdir('screenshots/croppedDomino'), trainSampleSize):
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/trainSplit/domino', file)
            shutil.copy(os.path.join('screenshots/croppedDomino', file), filePathToSave)
    for file in random.sample(os.listdir('screenshots/croppedNoDomino'), trainSampleSize):
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/trainSplit/noDomino', file)
            shutil.copy(os.path.join('screenshots/croppedNoDomino', file), filePathToSave)


if __name__ == "__main__":

    #prepare()
    trainModel1(train=True, epochs=1, folds=5)

 