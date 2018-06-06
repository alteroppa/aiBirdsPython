from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import utils
import os

imageWidth, imageHeight = 340, 125

def fetchImagesFromPool(train, test):
    utils.fetchImagesFromPool(1, 1+train, 'resizedNoDominoStructure', 'train/noDomino')
    utils.fetchImagesFromPool(1, 1+train, 'resizedDominoStructure', 'train/domino')
    utils.fetchImagesFromPool(1+train, 1+train+test, 'resizedNoDominoStructure', 'test/noDomino')
    utils.fetchImagesFromPool(1+train, 1+train+test, 'resizedDominoStructure', 'test/domino')

def emptyTrainAndTestFolders():
    utils.emptyFolder('train/noDomino')
    utils.emptyFolder('train/domino')
    utils.emptyFolder('test/noDomino')
    utils.emptyFolder('test/domino')

def trainModel1(train, epochs=5):
    print('Training model with', epochs, 'epochs...')
    imgWidth, imgHeight = imageWidth, imageHeight

    trainDataDir = 'train'
    testDataDir = 'test'

    trainSamples = 800
    testSamples = 200

    epochs = epochs #should be more like 50
    batchSize = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (3, imgWidth, imgHeight)
    else:
        input_shape = (imgWidth, imgHeight, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    '''
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))'''

    model.add(Flatten())
    model.add(Dense(units = 128, activation ='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

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
    A DirectoryIterator yielding tuples of(x, y) where x is a numpy array containing a batch of images with shape(batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
    '''

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

    if train:
        model.fit_generator(
            trainingSet,
            steps_per_epoch=trainSamples // batchSize,
            epochs=epochs,
            validation_data=testSet,
            validation_steps=testSamples // batchSize,
            verbose=1)

        model.save_weights('firstTry.h5')
    else:
        model.load_weights('firstTry.h5')


    validateImages(model, trainingSet)

def trainModel2():
    return


def validateImages(model, trainingSet):
    print(trainingSet.class_indices)
    for file in os.listdir('validationFolder'):
        if not file.startswith('.'):
            utils.resizeImagesInFolder((imageWidth, imageHeight), 'validationFolder')
            filePlusDir = os.path.join('validationFolder', file)
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

if __name__ == "__main__":
    #emptyTrainAndTestFolders()
    #utils.emptyFolder('resizedDominoStructure')
    #utils.emptyFolder('resizedNoDominoStructure')
    #utils.resizeImages((imageWidth, imageHeight)) # has to be a tuple, has to maintain aspect ratio of 1,75
    #fetchImagesFromPool(400, 100)
    trainModel1(train=False, epochs=20)
    #trainModel2()



    #print('domino:',utils.getFolderSize('originalDominoStructure'))
    #print('no domino:',utils.getFolderSize('originalNoDominoStructure'))