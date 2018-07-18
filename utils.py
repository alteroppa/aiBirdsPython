from PIL import Image
import os
import shutil

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

def getCoordinatesFile():
    coordinatesSrc = '../aiBirds/levelGenerator/generatedLevels/dominoCoordinates.txt'
    coordinatesGoal = 'screenshots/coordinates.txt'
    shutil.copy(coordinatesSrc, coordinatesGoal)


def fetchImagesFromSourceToOriginalFolders():
    print('fetching images from source (aibirds/abV1.32/screenshots) to screenshots/original[No]Domino ...')
    srcfolderDomino = '../aiBirds/abV1.32/screenshots/dominoStructure'
    goalfolderDomino = 'screenshots/originalDomino'
    amountOfPicsDomino = len(os.listdir(srcfolderDomino))
    fetchImagesFromPool(0, amountOfPicsDomino, srcfolderDomino, goalfolderDomino)
    srcfolderNoDomino = '../aiBirds/abV1.32/screenshots/noDominoStructure'
    goalfolderNoDomino = 'screenshots/originalNoDomino'
    amountOfPicsNoDomino = len(os.listdir(srcfolderNoDomino))
    fetchImagesFromPool(0, amountOfPicsNoDomino, srcfolderNoDomino, goalfolderNoDomino)

def resizeImagesInFolder(size, folderName):
    print('resizing images in folder ' + folderName + ' to size ' + str(size) + ' ...')
    for file in os.listdir(folderName):
        if file.endswith('.png'):
            image = Image.open(folderName + '/' + file)
            newImage = image.resize(size)
            newImage.save(folderName +'/' + file)

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

            screenshotWithDominoDone = False
            for j in range(4):
                x1 = (j) * quarterWidth
                x2 = (j+1) * quarterWidth

                # top half cut in 4 quarters
                newImageTopHalf = newImage.crop((x1, 0, x2, halfHeight))

                # for now: don't save top half, because there is no point anyway - it's just blue sky. Enable optionally.
                #newImageTopHalf.save('screenshots/croppedNoDomino/croppedTop' + file[:-4] + '_' + str(j+1) + '.png')

                # bottom half cut in 4 quarters
                newImageBottomHalf = newImage.crop((x1, halfHeight, x2, newHeight))
                if structureX > int((x1 + quarterWidth/8)) and structureX < int((x2 - quarterWidth/8)):
                    #print('structurex: ', structureX)
                    #print('x1: ', (x1 + quarterWidth/9))
                    #print('x2: ', (x2 - quarterWidth/9))
                    newImageBottomHalf.save('screenshots/croppedDomino/croppedBot' + file[:-4] + '_' + str(j+1) + '.png')
                    screenshotWithDominoDone = True
                else:
                    newImageBottomHalf.save('screenshots/croppedNoDomino/croppedBot' + file[:-4] + '_' + str(j+1) + '.png')

            # if no screenshot with domino done yet, do one extra
            '''if screenshotWithDominoDone == False:
                definitiveDominoStructure = newImage.crop((structureX - quarterWidth/2, halfHeight, structureX + quarterWidth/2, newHeight))
                definitiveDominoStructure.save(
                    'screenshots/croppedDomino/croppedBot' + file[:-4] + '_extra.png')'''

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
