import cv2
import numpy
import math


class Image:
    DELTA = .05

    def __init__(self, filePath):
        self.filePath = filePath
        self.src = cv2.imread(filePath)
        self.normalizedImg = self.normalize()
        self.countours = self.getContours()
        self.geometries = self.getGeometries()
        self.checkGeometries = self.checkGeometries()
        print(self.checkGeometries)
        self.drawContouredImg = self.drawContours()

    def normalize(self):
        '''Нормализация изображения'''

        grayImg = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        blurImg = cv2.medianBlur(grayImg, 5)
        canny = cv2.Canny(blurImg, 10, 250)

        return canny

    def getContours(self):
        '''Возвращает контуры'''

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(
            self.normalizedImg, cv2.MORPH_CLOSE, structuringElement)

        countoursReturn = []
        contours = cv2.findContours(
            closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cont in contours:
            sm = cv2.arcLength(cont, True)
            apd = cv2.approxPolyDP(cont, 0.01*sm, True)

            countoursReturn.append(apd)

        return countoursReturn

    def getGeometries(self):
        '''Возвращает длинны и диагонали'''

        contours = self.countours
        geometries = []

        for cont in contours:
            angles, coord, x = cont.shape

            # Если 4 угла
            if angles == 4:
                '''
                0 3
                1 2
                '''
                # Стороны
                s1 = self.getLenghtSegment([cont[0], cont[1]])
                s2 = self.getLenghtSegment([cont[1], cont[2]])
                s3 = self.getLenghtSegment([cont[2], cont[3]])
                s4 = self.getLenghtSegment([cont[3], cont[0]])
                # Диагонали
                d1 = self.getLenghtSegment([cont[0], cont[2]])
                d2 = self.getLenghtSegment([cont[1], cont[3]])

                geometries.append({
                    's1': s1, 's2': s2,
                    's3': s3, 's4': s4,
                    'd1': d1, 'd2': d2
                })
            else:
                geometries.append(False)

        return geometries

    def checkGeometries(self):
        '''Проверить геометрию'''

        checked = []
        geometries = self.geometries

        for geom in geometries:
            if geom == False:
                checked.append(False)
                continue

            checkSides1 = self.checkSide(geom.get('s1'), geom.get('s3'))
            checkSides2 = self.checkSide(geom.get('s2'), geom.get('s4'))
            checkDiag = self.checkSide(geom.get('d1'), geom.get('d2'))

            if (checkSides1 and checkSides2 and checkDiag):
                checked.append(True)
            else:
                checked.append(False)

        return checked

    def checkSide(self, side1, side2):
        '''Проверить равенство сторон с учетом допусков'''

        if (side1 <= side2 + side2 * self.DELTA or side1 >= side2 - side2 * self.DELTA):
            return True

        return False

    def getLenghtSegment(self, coord):
        '''Длинна отрезка из координат'''

        d = math.sqrt((coord[1][0][0] - coord[0][0][0]) ** 2 + (coord[1][0][1] - coord[0][0][1]) ** 2)
       
        return d

    def drawContours(self):
        '''Возвращает изображение с контурами'''

        img = self.src.copy()
        contours = self.countours

        for cont in contours:
            cv2.drawContours(img, [cont], -1, (0, 255, 0), 2)

        return img

    def showContouredImg(self):
        '''Показывает изображение с контурами'''

        cv2.imshow(self.filePath, self.drawContouredImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
