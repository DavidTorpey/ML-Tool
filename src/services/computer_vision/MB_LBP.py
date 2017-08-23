import cv2
import numpy as np

# define size of resizedImage
imageSize = (200,200)

def resize(image):
    return cv2.resize(image, imageSize)

image = resize(cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE))

def MB_LBP(image):
    ## define window size
    window_size = 3
    
    ## define the size of the mean image
    meanImage = np.zeros(imageSize)
    
    for x in xrange(image.shape[0]):
        for y in xrange(image.shape[1]):
            rect = image[x:x+window_size, y:y+window_size]
            meanBlock = int(rect.mean())
            meanImage[x][y] = meanBlock
    
    firstColumn = [column[0] for column in meanImage]
    lastColumn = [column[meanImage.shape[1]-1] for column in meanImage]
    firstRow = meanImage[:1]
    lastRow = meanImage[-1:]
    
    imageRows = imageSize[0]
    imageColumns = imageSize[1]
    
    # Cater for boundary case
    boundaryCase = np.zeros((imageRows + 2,imageColumns + 2))
    r,c = boundaryCase.shape
    boundaryCase[0, :][1:-1] = firstRow
    boundaryCase[c-1,:][1:-1] = lastRow
    boundaryCase[:,0][1:-1] = firstColumn
    boundaryCase[:,r-1][1:-1] = lastColumn
    boundaryCase[1:-1,1:-1] = meanImage
    
    LBP = np.zeros(imageSize)
    rows, columns = LBP.shape
    for x in xrange(rows):
        for y in xrange(columns):
            pixelWindow = []
            binNumber = '0'
            centerPixel = boundaryCase[x][y]
            topleft = boundaryCase[x-1][y-1]
            pixelWindow.append(topleft)
            
            top = boundaryCase[x-1][y]
            pixelWindow.append(top)
            
            topright = boundaryCase[x-1][y+1]
            pixelWindow.append(topright)
            
            left = boundaryCase[x][y-1]
            pixelWindow.append(left)
            
            right = boundaryCase[x][y+1]
            pixelWindow.append(right)
            
            bottomleft = boundaryCase[x+1][y-1]
            pixelWindow.append(bottomleft)
            
            bottom = boundaryCase[x+1][y]
            pixelWindow.append(bottom)
            
            bottomRight = boundaryCase[x+1][y+1]
            pixelWindow.append(bottomRight)
            
            for pix in pixelWindow:
                if centerPixel >= pix:
                    binNumber = binNumber + str(0)
                else:
                    binNumber = binNumber + str(1)
            
            decimalNumber = int(binNumber, 2)
            LBP[x][y] = decimalNumber
    
    return LBP