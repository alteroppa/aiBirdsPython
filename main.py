from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras import backend as K
from keras import callbacks
import shutil
import datetime
import numpy as np
import utils
import os
from statistics import mean, variance, stdev


'''
This file provides all necessary methods to run the training. All secondary, helper methods can be found in utils.py.
'''



#imageWidth, imageHeight = 187, 150
imageWidth, imageHeight = 93, 75
lowImageWith, lowImageHeight = 50, 40

'''
main method in this file. Defines all necessary parameters for the model and for documentation, gets a model, 
trains it and calls the validation method. Measures runtime, too.
'''
def trainModel(train, epochs=30, folds=5, cleanData=True, equalAmount=True):
    if cleanData:
        amountCroppedSplitDomino = int(len(os.listdir('screenshots/croppedDominoCleanOhne10'))) - 1
        if equalAmount:
            amountCroppedSplitNoDomino = amountCroppedSplitDomino
        else:
            amountCroppedSplitNoDomino = len(os.listdir('screenshots/croppedNoDominoCleanOhne10')) - 1

    else:
        amountCroppedSplitDomino = int(len(os.listdir('screenshots/croppedDominoFaultyOhne10'))) - 1
        if equalAmount:
            amountCroppedSplitNoDomino = amountCroppedSplitDomino
        else:
            amountCroppedSplitNoDomino = len(os.listdir('screenshots/croppedNoDominoFaultyOhne10')) - 1

    testAmountDomino = int(amountCroppedSplitDomino / 5)
    trainAmountDomino = int((amountCroppedSplitDomino / 5) * 4)
    testAmountNoDomino = int(amountCroppedSplitNoDomino / 5)
    trainAmountNoDomino = int((amountCroppedSplitNoDomino / 5) * 4)

    testAmount = testAmountDomino + testAmountNoDomino
    trainAmount = trainAmountDomino + trainAmountNoDomino
    print('domino total:',amountCroppedSplitDomino,'noDomino total:',amountCroppedSplitNoDomino)

    print('Training model with', epochs, 'epochs,', folds, ' folds...')
    if cleanData:
        print('used Data: clean')
    else:
        print('used Data: dirty')

    absoluteStartTime = datetime.datetime.now()
    imgWidth, imgHeight = imageWidth, imageHeight
    #imgWidth, imgHeight = lowImageWith, lowImageHeight

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

    meanAcc =0
    meanValAcc = 0
    meanValLoss = 0
    meanTrainLoss = 0
    bestAcc = 0
    listValAcc = []
    listValLoss = []
    listAcc = []

    # callbacks
    csv_logger = callbacks.CSVLogger('training' + str(datetime.datetime.now()) +'.log')
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.03, patience=4, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint('bestWeightsModel.h5', monitor='val_acc', verbose=1,
                                               save_best_only=True, mode='auto')
    history = callbacks.History()


    for i in range(folds):
        bestValAcc = 0
        lowestValLoss = 50
        print('training fold', i, '...')

        splitPrepare(amountCroppedSplitDomino, amountCroppedSplitNoDomino, clean=cleanData, fractionOfTest=5)

        startTime = datetime.datetime.now()

        model = getModel(input_shape)

        trainDataGen = ImageDataGenerator(rescale=1./255)
        testDataGen = ImageDataGenerator(rescale=1./255)

        trainingSet = trainDataGen.flow_from_directory(
            trainDataDir,
            target_size=(imgWidth, imgHeight),
            batch_size=batchSize,
            class_mode='binary')

        testSet = testDataGen.flow_from_directory(
            testDataDir,
            target_size=(imgWidth, imgHeight),
            batch_size=batchSize,
            class_mode='binary')

        if train:
            print('training...')
            history = model.fit_generator(
                trainingSet,
                steps_per_epoch=trainSamples // batchSize,
                epochs=epochs,
                validation_data=testSet,
                validation_steps=testSamples // batchSize,
                verbose=2, callbacks=[csv_logger, history, saveBestModel, earlyStopping])

            # reload best weights
            model.load_weights('bestWeightsModel.h5')
            #model.save_weights(str(datetime.datetime.now()) + 'MODEL.h5')
        else: # in case you only want to predict, not train
            model.load_weights('23:20:23.h5')
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            validateImages(model)

        utils.plot(history, epochs, trainSamples, testSamples)

        endtime = datetime.datetime.now()
        duration = endtime - startTime
        print("duration of epoch: ", str(duration))

        if max(history.history['val_acc']) > bestValAcc:
            bestValAcc = max(history.history['val_acc'])
        if min(history.history['val_loss']) < lowestValLoss:
            lowestValLoss = min(history.history['val_loss'])
        if max(history.history['acc']) > bestAcc:
            bestAcc = max(history.history['acc'])

        # these values are measured each epoch
        val_acc = max(history.history['val_acc']) #ensures that the best epoch (regarding val acc) is used for measurements
        index = history.history['val_acc'].index(val_acc)
        loss = history.history['loss'][index] # takes the values on index[val_acc]
        val_loss = history.history['val_loss'][index]
        acc = history.history['acc'][index]
        listValAcc.append(val_acc)
        listValLoss.append(val_loss)
        listAcc.append(acc)

        print('best val_acc:', round(bestValAcc, 4))
        print('lowest val_loss:', round(lowestValLoss, 4))
        print('best acc:', round(bestAcc, 4))




        if i == 0:
            utils.latexTableTopline()
        utils.latexTable(trainAmount, testAmount, epochs, history)
        if i == folds-1:
            overallValAccStdev = stdev(listValAcc)
            overallValLossStdev = stdev(listValLoss)
            absoluteDuration = datetime.datetime.now() - absoluteStartTime
            utils.latexTableBottomline(absoluteDuration, bestValAcc, overallValAccStdev, overallValLossStdev)


        if cleanData:
            validateImages(model, 'screenshots/validationFolderClean')
        else:
            validateImages(model, 'screenshots/validationFolderFaulty')


        #validateImages(model, 'screenshots/realLevelsSplit')

    meanTrainLoss = meanTrainLoss/folds
    meanValAcc = meanValAcc/folds
    meanValLoss = meanValLoss/folds
    meanAcc = meanAcc/folds
    valAccStd = stdev(listValAcc)
    valLossStd = stdev(listValLoss)
    accStd = stdev(listAcc)

    # these values are measured over ALL epochs
    print('acc std:', round(accStd,4))
    print('val acc std:', round(valAccStd,4))
    print('val loss std:', round(valLossStd,4))
    print('mean acc:', round(meanAcc,4))
    print('meanTrainLoss:',round(meanTrainLoss,4))
    print('meanValLoss:',round(meanValLoss,4))
    print('meanValAcc:',round(meanValAcc,4))




'''
creates a model. The Dropout can be activated, though I didn't see much difference.
'''
def getModel(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5)) #

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5)) #

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


'''
predicts all screenshots in a specific folder. Those screenshots must be the same size as the training data. 
'''
def validateImages(model, validationDir):
    for file in os.listdir(validationDir):
        if not file.startswith('.'):
            filePlusDir = os.path.join(validationDir, file)
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


'''
function used for preparation. Fetches files from original folders of where the java agent saved the created screenshots.
CARE: those functions in utils use relative paths that will likely not be present on your machine! Change accordingly.

This function doesn't have to be called in every run. 
Also, only parts of the function can be called to fulfil your needs. Just uncomment the lines accordingly.
'''
def prepare():
    #utils.emptyTrainAndTestFolders()
    #utils.emptyFolder('screenshots/resizedDomino')
    #utils.emptyFolder('screenshots/resizedNoDomino')
    #utils.emptyFolder('screenshots/originalDomino')
    #utils.emptyFolder('screenshots/originalNoDomino')
    #utils.emptyFolder('screenshots/validationFolder')
    #utils.emptyFolder('screenshots/croppedDomino')
    #utils.emptyFolder('screenshots/croppedNoDomino')
    #utils.getCoordinatesFile()
    #utils.fetchImagesFromSourceToOriginalFolders()
    #utils.splitAndCropImagesSlidingWindow('screenshots/originalDomino')
    #utils.splitAndCropImagesSlidingWindow('screenshots/originalNoDomino')



def splitPrepare(amountCroppedSplitDomino, amountCroppedSplitNoDomino, clean=True, fractionOfTest=5):
    utils.emptyFolder('screenshots/testSplit/domino')
    utils.emptyFolder('screenshots/testSplit/noDomino')
    utils.emptyFolder('screenshots/trainSplit/domino')
    utils.emptyFolder('screenshots/trainSplit/noDomino')

    if clean: # checks if you want to train on the by-hand cleaned data or on the 'dirty' data
        dominoFolder='screenshots/croppedDominoCleanOhne10'
        noDominoFolder = 'screenshots/croppedNoDominoCleanOhne10'
    else:
        dominoFolder='screenshots/croppedDominoFaultyOhne10'
        noDominoFolder = 'screenshots/croppedNoDominoFaultyOhne10'

    randomDomino = np.random.choice(amountCroppedSplitDomino, amountCroppedSplitDomino, replace=False).tolist()
    finalTestAmountDomino = int(amountCroppedSplitDomino / fractionOfTest)
    randomDominoTest = randomDomino[:finalTestAmountDomino]
    randomDominoTrain = randomDomino[finalTestAmountDomino:]

    print('length of randomDomino:', len(randomDomino))
    print('randomDominoTest:', randomDominoTest[:5], ' lenght:', len(randomDominoTest))
    print('randomDominoTrain:', randomDominoTrain[:5], ' lenght:', len(randomDominoTrain))


    randomNoDomino = np.random.choice(amountCroppedSplitNoDomino, amountCroppedSplitNoDomino, replace=False).tolist()
    finalTestAmountNoDomino = int(amountCroppedSplitNoDomino / fractionOfTest)
    randomNoDominoTest = randomNoDomino[:finalTestAmountNoDomino]
    randomNoDominoTrain = randomNoDomino[finalTestAmountNoDomino:]

    print('length of randomNoDomino:', len(randomNoDomino))
    print('randomNoDominoTest:', randomNoDominoTest[:5], ' lenght:', len(randomNoDominoTest))
    print('randomNoDominoTest:', randomNoDominoTrain[:5], ' lenght:', len(randomNoDominoTrain))


    print('now copying to testSplit/domino ...')
    for i in randomDominoTest:
        file = os.listdir(dominoFolder)[i]
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/testSplit/domino', file)
            shutil.copy(os.path.join(dominoFolder, file), filePathToSave)

    print('now copying to trainSplit/domino ...')
    for i in randomDominoTrain:
        file = os.listdir(dominoFolder)[i]
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/trainSplit/domino', file)
            shutil.copy(os.path.join(dominoFolder, file), filePathToSave)

    print('now copying to testSplit/noDomino ...')
    for i in randomNoDominoTest:
        file = os.listdir(noDominoFolder)[i]
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/testSplit/noDomino', file)
            shutil.copy(os.path.join(noDominoFolder, file), filePathToSave)

    print('now copying to trainSplit/noDomino ...')
    for i in randomNoDominoTrain:
        file = os.listdir(noDominoFolder)[i]
        if file.endswith('.png'):
            filePathToSave = os.path.join('screenshots/trainSplit/noDomino', file)
            shutil.copy(os.path.join(noDominoFolder, file), filePathToSave)



if __name__ == "__main__":
    startTime = datetime.datetime.now()
    #prepare() # does not have to be called every run, rather only once to clean up and (re)-fetch the screenshots
    trainModel(train=True, epochs=30, folds=5, cleanData=True, equalAmount=True)
    endtime = datetime.datetime.now()
    delta = endtime - startTime
    print("total duration over 5 folds:", str(delta))
    #utils.copyToErrorFolder()

 