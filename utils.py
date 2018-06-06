from PIL import Image
import os
import shutil

def resizeImages(size):
    for file in os.listdir('originalDominoStructure'):
        if not file.startswith('.'):
            image = Image.open('originalDominoStructure/' + file)
            newImage = image.resize(size)
            newImage.save('resizedDominoStructure/' + file)
    for file in os.listdir('originalNoDominoStructure'):
        if not file.startswith('.'):
            image = Image.open('originalNoDominoStructure/' + file)
            newImage = image.resize(size)
            newImage.save('resizedNoDominoStructure/' + file)

def resizeImagesInFolder(size, folderName):
    for file in os.listdir(folderName):
        if not file.startswith('.'):
            image = Image.open(folderName + '/' + file)
            newImage = image.resize(size)
            newImage.save(folderName +'/' + file)

def getFolderSize(folderName):
    return(len(os.listdir(folderName)))

def fetchImagesFromPool(amountFrom, amountTo, folderFrom, folderTo):
    srcFiles = os.listdir(folderFrom)[amountFrom:amountTo]
    for fileName in srcFiles:
        if not fileName.startswith('.'):
            fullFileNameFrom = os.path.join(folderFrom, fileName)
            fullFileNameTo = os.path.join(folderTo, fileName)
            shutil.copy(fullFileNameFrom, fullFileNameTo)

def emptyFolder(folderName):
    for file in os.listdir(folderName):
        os.remove(os.path.join(folderName,file))

def cropImagesInFolder(folderName):
    for file in os.listdir(folderName):
        if not file.startswith('.'):
            image = Image.open(folderName + '/' + file)
            width, height = image.size
            newImage = image.crop((0, 40, width-10, 165))
            newImage.save(folderName + '/' + file)