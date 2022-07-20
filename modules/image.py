import cv2
import numpy


class Image:
    def __init__(self, filePath):
        self.filePath = filePath
        self.src = cv2.imread(filePath)
        self.normalizedImg = self.normalize()
        self.drawContouredImg = self.drawContours()

    def normalize(self):
        grayImg = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        blurImg = cv2.medianBlur(grayImg, 5)
        canny = cv2.Canny(blurImg, 10, 250)

        return canny

    def drawContours(self):
        img = self.src.copy()
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(
            self.normalizedImg, cv2.MORPH_CLOSE, structuringElement)

        contours = cv2.findContours(
            closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cont in contours:
            sm = cv2.arcLength(cont, True)
            apd = cv2.approxPolyDP(cont, 0.01*sm, True)

            cv2.drawContours(img, [apd], -1, (0, 255, 0), 2)

        return img

    def showContouredImg(self):
        cv2.imshow(self.filePath, self.drawContouredImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
