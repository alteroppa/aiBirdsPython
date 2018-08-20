from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import binom
import scipy.stats
from statistics import mean, stdev

# the values of the last epoch each
listAcc = []
listValAcc= []
listLoss = []
listValLoss=[]


def resizeImages(size):
    print('resizing images in original folders to size ' + size + ' ...')
    for file in os.listdir('screenshots/originalDomino'):
        if file.endswith('.png'):
            image = Image.open('screenshots/originalDomino/' + file)
            newImage = image.resize(size)
            newImage.save('screenshots/resizedDomino/' + file)
    for file in os.listdir('screenshots/originalNoDomino'):
        if file.endswith('.png'):
            image = Image.open('screenshots/originalNoDomino/' + file)
            newImage = image.resize(size)
            newImage.save('screenshots/resizedNoDomino/' + file)

def resizeImagesDirectlyInFolder(size, folder):
    print('resizing images in', folder, ' to size ' + size + ' ...')
    for file in os.listdir(folder):
        if file.endswith('.png'):
            image = Image.open(folder + '/' + file)
            newImage = image.resize(size)
            newImage.save(folder + '/' + file)

def getCoordinatesFile():
    coordinatesSrc = '../aiBirds/levelGenerator/generatedLevels/dominoCoordinates.txt'
    coordinatesGoal = 'screenshots/coordinates.txt'
    shutil.copy(coordinatesSrc, coordinatesGoal)

def fetchAllImagesFromPool(train, test):
    fetchImagesFromPool(1, 1+train, 'screenshots/croppedNoDomino', 'screenshots/train/noDomino')
    fetchImagesFromPool(1, 1+train, 'screenshots/croppedDomino', 'screenshots/train/domino')
    fetchImagesFromPool(1+train, 1+train+test, 'screenshots/croppedNoDomino', 'screenshots/test/noDomino')
    fetchImagesFromPool(1+train, 1+train+test, 'screenshots/croppedDomino', 'screenshots/test/domino')

def emptyTrainAndTestFolders():
    emptyFolder('screenshots/train/noDomino')
    emptyFolder('screenshots/train/domino')
    emptyFolder('screenshots/test/noDomino')
    emptyFolder('screenshots/test/domino')

def fetchImagesFromSourceToOriginalFolders():
    print('fetching images from source (aibirds/abV1.32/screenshots [LOCAL SOURCE!]) to screenshots/original[No]Domino ...')
    srcfolderDomino = '../aiBirds/abV1.32/screenshots/dominoStructure'
    goalfolderDomino = 'screenshots/originalDomino'
    amountOfPicsDomino = len(os.listdir(srcfolderDomino))
    fetchImagesFromPool(0, amountOfPicsDomino, srcfolderDomino, goalfolderDomino)
    srcfolderNoDomino = '../aiBirds/abV1.32/screenshots/noDominoStructure'
    goalfolderNoDomino = 'screenshots/originalNoDomino'
    amountOfPicsNoDomino = len(os.listdir(srcfolderNoDomino))
    fetchImagesFromPool(0, amountOfPicsNoDomino, srcfolderNoDomino, goalfolderNoDomino)

def getFolderSize(folderName):
    return(len(os.listdir(folderName)))

def fetchImagesFromPool(amountFrom, amountTo, folderFrom, folderTo):
    print('fetching images from pool from folder ' + folderFrom + ' to folder ' + folderTo + ' from amount ' + str(amountFrom) + ' to amount ' + str(amountTo) + ' ...')
    srcFiles = os.listdir(folderFrom)[amountFrom:amountTo]
    amount = 0
    for fileName in srcFiles:
        if fileName.endswith('.png'):
            fullFileNameFrom = os.path.join(folderFrom, fileName)
            fullFileNameTo = os.path.join(folderTo, fileName)
            shutil.copy(fullFileNameFrom, fullFileNameTo)
            amount = amount + 1
    print('fetched ' + str(amount) + ' pictures from ' + folderFrom + ' and saved them at ' + folderTo)



def emptyFolder(folderName):
    amount = len(os.listdir(folderName))
    for file in os.listdir(folderName):
        os.remove(os.path.join(folderName,file))
    print('emptied folder ' + folderName + ' with ' + str(amount) + 'files.')

def cropImagesInFolder(folderName):
    print('cropping images in folder ' + folderName + '...')
    for file in os.listdir(folderName):
        if file.endswith('.png'):
            image = Image.open(folderName + '/' + file)
            width, height = image.size
            newImage = image.crop((0, 40, width-10, 165))
            newImage.save(folderName + '/' + file)


def splitAndCropImagesSlidingWindow(folderToRead):
    print('splitting and cropping images in folder ' + folderToRead + '...')
    with open('screenshots/coordinates.txt') as coordsFromFile:
        coordinates = coordsFromFile.readlines()
    cutPixelsUntilSlingshot = 52
    cutPixelsFromEnd = 40

    for file in os.listdir(folderToRead):
        if file.endswith('.png'):
            structureX = 0
            distance = 0
            if not 'realLevels' in folderToRead:
                levelNumber = file[5:-4]
                structureX = (int(coordinates[int(levelNumber) - 1].split(';')[1])) * 5 + 14 - cutPixelsUntilSlingshot
                distance = (int(coordinates[int(levelNumber) - 1].split(';')[3])) * 5 + 14
            image = Image.open(folderToRead + '/' + file)
            originalWidth, originalHeight = image.size
            newImage = image.crop((cutPixelsUntilSlingshot, 90, originalWidth - cutPixelsFromEnd, 390))
            newWidth, newHeight = newImage.size
            quarterWidth = newWidth/4
            halfHeight = newHeight/2

            windowEndWidth = quarterWidth
            windowStartWidth = 0
            iterator = 0

            while windowEndWidth <= newWidth:
                windowBottom = newImage.crop((windowStartWidth, halfHeight, windowEndWidth, newHeight))
                if 'realLevels' in folderToRead:
                    windowBottom.save('screenshots/realLevelsSplit/realLvl' + file[:-4] + '_' + str(iterator + 1) + '.png')
                if 'originalDomino' in folderToRead:
                    if structureX > int((windowStartWidth + distance)) and structureX < int((windowEndWidth - distance)):
                        windowBottom.save('screenshots/croppedDomino/cropDomino' + file[:-4] + '_' + str(iterator+1) + '.png')
                    # to make sure structure coordinates are OUT of the window, deduct/add 30px as a threshold:
                    if structureX < int((windowStartWidth + distance - 30)) or structureX > int((windowEndWidth - distance + 30)):
                        windowBottom.save('screenshots/croppedNoDomino/cropNoDomino' + file[:-4] + '_' + str(iterator+1) + '.png')
                if 'originalNoDomino' in folderToRead:
                    windowBottom.save('screenshots/croppedNoDomino/cropNoDomino' + file[:-4] + '_' + str(iterator + 1) + '.png')

                windowStartWidth += 20
                windowEndWidth += 20
                iterator += 1




def splitAndCropImagesDominoScreenshots(folderToRead):
    print('splitting and cropping images in folder ' + folderToRead + '...')
    with open('screenshots/coordinates.txt') as coordsFromFile:
        coordinates = coordsFromFile.readlines()
    cutPixelsUntilSlingshot = 52
    cutPixelsFromEnd = 40

    for file in os.listdir(folderToRead):
        if file.endswith('.png'):
            levelNumber = file[5:-4]
            #structureX = (float(coordinates[levelCounter].split(';')[1])) * 5 + 14 - cutPixelsUntilSlingshot
            structureX = (int(coordinates[int(levelNumber) - 1].split(';')[1])) * 5 + 14 - cutPixelsUntilSlingshot
            #structureY =  #coordinates[int(levelNumber) -1].split(';')[2]

            image = Image.open(folderToRead + '/' + file)
            originalWidth, originalHeight = image.size
            newImage = image.crop((cutPixelsUntilSlingshot, 90, originalWidth - cutPixelsFromEnd, 390))
            #newImage.save('screenshots/croppedDomino/croppedOriginal' + file[:-4] + '_' + str(i) + '.png')
            newWidth, newHeight = newImage.size

            quarterWidth = newWidth/4
            halfHeight = newHeight/2

            for j in range(4):
                x1 = (j) * quarterWidth
                x2 = (j+1) * quarterWidth

                # bottom half cut in 4 quarters
                newImageBottomHalf = newImage.crop((x1, halfHeight, x2, newHeight))
                if structureX > int((x1 + quarterWidth/8)) and structureX < int((x2 - quarterWidth/8)):
                    newImageBottomHalf.save('screenshots/croppedDomino/cropDomino' + file[:-4] + '_' + str(j+1) + '.png')
                    screenshotWithDominoDone = True
                else:
                    newImageBottomHalf.save('screenshots/croppedNoDomino/cropNoDomino' + file[:-4] + '_' + str(j+1) + '.png')


def plot(history, totalEpochs, trainSamples, testSamples):
    print(history.history)
    #history.history is a dict with 'val_loss'=[...], 'val_acc'=..., 'val', 'loss'
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']


    bestValAcc = str(history.history['val_acc'][-1:])
    print('best val acc:', bestValAcc)
    with open ('latexTable.txt', mode='a') as latexfile:
        latexfile.write('\nbest val acc: '+bestValAcc)

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss',)
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training und validation loss (' + str(trainSamples) + '/' + str(testSamples) + ' train/test samples)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(totalEpochs) + ' epcs, ' + str(trainSamples) + 'trnSmpls ' + str(datetime.now()) + '.png')
    plt.show()

def latexTableTopline():
    with open('latexTable.txt', mode='a') as latexfile:
        latexfile.write('\n\n' + str(datetime.now()) +
                        '\n\\begin{table}[]\n\centering{}\n\\resizebox{\\textwidth}{!}{%'
                        '\n\\begin{tabular}{lllllllll}\n'
                        + '\#.fold & \(N_{train}\)/\(N_{test}\) '
                        + '& \\textit{dom:noDom} & \\textit{epochs} & \\textit{acc} '
                          ' & \\textit{val loss} & \\textit{val acc} \\\\\n\hline')

def latexTable(trainAmount, testAmount, epochs, history):
    listAcc.append(history.history['acc'][-1:][0])
    listValAcc.append(history.history['val_acc'][-1:][0])
    listLoss.append(history.history['loss'][-1:][0])
    listValLoss.append(history.history['val_loss'][-1:][0])
    with open('latexTable.txt', mode='a') as latexfile:
        latexfile.write('\n'
                        + 'run.fold' + ' & '
                        + str(trainAmount) + '/' + str(testAmount) + ' & '
                        + '1:1' + ' & '
                        + str(epochs) + ' & '
                        + str("%.4f" % round(history.history['acc'][-1:][0], 4)) + ' & '
                        + str("%.4f" % round(history.history['val_loss'][-1:][0],4)) + ' & '
                        + str("%.4f" % round(history.history['val_acc'][-1:][0],4)) + '\\\\'
                        )

def latexTableBottomline(duration, bestValAcc, overallValStdev, overallValLossStdev):
    with open('latexTable.txt', mode='a') as latexfile:
        latexfile.write('\nMittelwert & - & - & - & ' + str(round(mean(listAcc),4)) + ' & ' + str(round(mean(listValLoss),4)) + ' & ' + str(round(mean(listValAcc),4)) + ' \\'
                        + '\nStandardabw. & - & - & - & ' + str(round(stdev(listAcc),4)) + ' & ' + str(round(stdev(listValLoss),4)) + ' & ' + str(round(stdev(listValAcc),4)) + ' \\'
                        + '\n\hline\n\end{tabular}%\n}\n\caption{Rechenzeit: ' + str(duration)
                        + ', beste val acc:' + str(round(bestValAcc, 4))
                        + 'val acc Stdev: ' + str(round(overallValStdev,4))
                        + 'val loss Stdev: ' + str(round(overallValLossStdev, 4))
                        + '}\n\end{table}')


def splitAndCropImagesNoDominoScreenshots(folderToRead):
    print('splitting and cropping images in folder ' + folderToRead + '...')
    cutPixelsUntilSlingshot = 52
    cutPixelsFromEnd = 40
    levelCounter = 0
    for file in os.listdir(folderToRead):
        if file.endswith('.png'):
            image = Image.open(folderToRead + '/' + file)
            originalWidth, originalHeight = image.size
            newImage = image.crop((cutPixelsUntilSlingshot, 90, originalWidth - cutPixelsFromEnd, 390))
            # newImage.save('screenshots/croppedDomino/croppedOriginal' + file[:-4] + '_' + str(i) + '.png')
            newWidth, newHeight = newImage.size

            quarterWidth = newWidth / 4
            halfHeight = newHeight / 2
            for j in range(4):
                x1 = (j) * quarterWidth
                x2 = (j + 1) * quarterWidth

                # top half cut in 4 quarters
                newImageTopHalf = newImage.crop((x1, 0, x2, halfHeight))

                # for now: don't save top half, because there is no point anyway - it's just blue sky. Enable optionally.
                # newImageTopHalf.save('screenshots/croppedNoDomino/croppedTop' + file[:-4] + '_' + str(j+1) + '.png')

                # bottom half cut in 4 quarters
                newImageBottomHalf = newImage.crop((x1, halfHeight, x2, newHeight))
                newImageBottomHalf.save('screenshots/croppedNoDomino/croppedBot' + file[:-4] + '_' + str(j + 1) + '.png')

def calcPercentile():
    data_binom = binom.rvs(n=120, p=1 / 120, size=20000)
    CI = binom.interval(0.95, 120, 1 / 120)
    print(CI)
    plt.hist(data_binom)
    plt.show()

def clopper_pearson(k=1,n=200,alpha=0.05):
    lowerBoundary = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    higherBoundary = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    print(lowerBoundary, higherBoundary)
