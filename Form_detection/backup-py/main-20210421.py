import cv2
import numpy as np
import pytesseract
import os
import imutils
import re

# C:\Program Files\Tesseract-OCR


class Form_detection:
    def __init__(self):
        self.template = r'D:\Desktop\Desktop_2\1.Python_project\PDF_Automation\image\logistic_template-2.jpg'
        self.target = r''

        self.per = 25
        self.pixelThreshold = 500
        self.scale = 0.5

        self.roi = [[(1896, 764), (2348, 808), 'date', 'INVOICE DATE'],
                    [(148, 1592), (868, 1636), 'text', 'IMPORT CUSTOMS BROKER'],
                    # [(140, 1968), (1884, 2764), 'table_field', 'DESCRIPTION_ITEM'],
                    # [(1996, 1976), (2380, 2768), 'table_value', 'CHARGES IN HKD'],
                    [(100, 1900), (2370, 2756), 'table', 'DESCRIPTION']
                    ]

        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        #config = r"-c tessedit_char_blacklist=—!$%«‘&\'*+-/;<=>?@[\\]^_`{|}~ --oem 3 --psm 3"
        self.config = r'-c tessedit_char_blacklist=«'

    def thresholding(self, img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)[1]
        return img

    def cleanup_text(self, text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()
        # for c in text:
        #     print(c,ord(c))
        # print('****************\n')

        # return "".join([c if ((ord(c) >= 65 and ord(c) <= 90)or(ord(c) >= 97 and ord(c) <= 122) or (ord(c) >= 48 and ord(c) <= 57)or ord(c)==32 or ord(c)==46 or ord(c)==10) else "" for c in text]).strip()

    def filter_text(self, ocr_text):
        ocr_text = self.cleanup_text(ocr_text)
        ocr_text = ocr_text.replace(re.search('[a-z]{1}[A-Z]{1}', ocr_text).group(
        ) if re.search(' [a-z]{1}[A-Z]{1} ', ocr_text)else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [a-z]{3} ', ocr_text).group() if re.search(' [a-z]{3} ', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(' [A-Z]{1}[a-z]{1} ', ocr_text).group(
        ) if re.search(' [A-Z]{1}[a-z]{1} ', ocr_text)else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(' [A-Z]{1}[a-z]{1}\n\n', ocr_text).group(
        ) if re.search(' [A-Z]{1}[a-z]{1}\n\n', ocr_text)else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [a-z]{2} ', ocr_text).group() if re.search(' [a-z]{2} ', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [a-z]{2}\n\n', ocr_text).group() if re.search(' [a-z]{2}\n\n', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [a-z]{2}\n', ocr_text).group() if re.search(' [a-z]{2}\n', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [a-z]{3}\n', ocr_text).group() if re.search(' [a-z]{3}\n', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [a-z]{1} ', ocr_text).group() if re.search(' [a-z]{1} ', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [a-z]{1}\n', ocr_text).group() if re.search(' [a-z]{1}\n', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [0-9]{1} ', ocr_text).group() if re.search(' [0-9]{1} ', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [0-9]{1}\n\n', ocr_text).group() if re.search(' [0-9]{1}\n\n', ocr_text) else 'abc', ' ')
        ocr_text = ocr_text.replace(re.search(
            ' [0-9]{1}\n', ocr_text).group() if re.search(' [0-9]{1}\n', ocr_text) else 'abc', ' ')

        return ocr_text

    def main(self):
        imgQ = cv2.imread(self.template)
        h, w, c = imgQ.shape
        #imgQ = cv2.resize(imgQ,(w//3,h//3))

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)
        # impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

        # cv2.imshow("KeyPointsQuery",impKp1)

        path = r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\UserForms'
        myPicList = os.listdir(path)
        print(myPicList)
        for j, y in enumerate(myPicList):
            img = cv2.imread(path + "/"+y)
            #img = cv2.resize(img, (w // 3, h // 3))
            #cv2.imshow(y, img)
            kp2, des2 = orb.detectAndCompute(img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(des2, des1)
            matches.sort(key=lambda x: x.distance)
            good = matches[:int(len(matches)*(self.per/100))]
            imgMatch = cv2.drawMatches(
                img, kp2, imgQ, kp1, good[:100], None, flags=2)

            # cv2.imshow(y, imgMatch)

            srcPoints = np.float32(
                [kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPoints = np.float32(
                [kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
            imgScan = cv2.warpPerspective(img, M, (w, h))
            # imgScan = cv2.resize(imgScan, (w // 3, h // 3))
            # cv2.imshow(y, imgScan)

            imgShow = imgScan.copy()
            imgMask = np.zeros_like(imgShow)

            myData = []

            print(
                f'################## Extracting Data from Form {j}  ##################')

            for x, r in enumerate(self.roi):
                cv2.rectangle(imgMask, (r[0][0], r[0][1]),
                              (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
                imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

                imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                # imgCrop=thresholding(imgCrop)
                hc, wc, cc = imgCrop.shape
                imgCrop = cv2.resize(
                    imgCrop, (int(wc*self.scale), int(hc*self.scale)))
                # imgCrop=thresholding(imgCrop)

                if r[2] == 'text':
                    # imgCrop=thresholding(imgCrop)
                    ocr_text = pytesseract.image_to_string(
                        imgCrop, config=self.config)
                    ocr_text = self.filter_text(ocr_text)
                    print('{} :\n{} \n**************'.format(r[3], (ocr_text)))

                    myData.append(ocr_text)

                if r[2] == 'date':
                    # imgCrop=thresholding(imgCrop)
                    ocr_text = pytesseract.image_to_string(
                        imgCrop, config=self.config)
                    ocr_text = self.filter_text(ocr_text)
                    print('{} :\n{}\n**************'.format(r[3], (ocr_text)))
                    myData.append(ocr_text)

                if r[2] == 'table':
                    # imgCrop=thresholding(imgCrop)
                    # blur = cv2.GaussianBlur(imgCrop, (3,3), 0)
                    # imgCrop = 255 - blur

                    ocr_text = pytesseract.image_to_string(
                        imgCrop, config=self.config)
                    ocr_text = self.filter_text(ocr_text)
                    print('{} :\n{}\n**************'.format(r[3], (ocr_text)))
                    myData.append(ocr_text)

                if r[2] == 'table_field':
                    # imgCrop=thresholding(imgCrop)
                    ocr_text = pytesseract.image_to_string(
                        imgCrop, config=self.config)
                    ocr_text = self.filter_text(ocr_text)
                    print('{} :\n{}\n**************'.format(r[3], (ocr_text)))
                    myData.append(ocr_text)

                if r[2] == 'table_value':
                    ocr_text = pytesseract.image_to_string(
                        imgCrop, config=self.config)
                    ocr_text = self.filter_text(ocr_text)
                    print('{} :\n{}\n**************'.format(r[3], (ocr_text)))
                    myData.append(ocr_text)

                if r[2] == 'box':
                    hc, wc, cc = imgCrop.shape
                    imgCrop = cv2.resize(
                        imgCrop, (int(wc*self.scale), int(hc*self.scale)))
                    imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                    imgThresh = cv2.threshold(
                        imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                    totalPixels = cv2.countNonZero(imgThresh)
                    if totalPixels > self.pixelThreshold:
                        totalPixels = 1
                    else:
                        totalPixels = 0
                    print(f'{r[3]} :\n {totalPixels}\n**************')
                    myData.append(totalPixels)

                if r[2] == '':
                    hc, wc, cc = imgCrop.shape
                    imgCrop = cv2.resize(
                        imgCrop, (int(wc*self.scale), int(hc*self.scale)))
                    imgCrop = self.thresholding(imgCrop)
                    ocr_text = pytesseract.image_to_string(
                        imgCrop, config=self.config)
                    ocr_text = self.filter_text(ocr_text)
                    print('{} :\n{}\n**************'.format(r[3], ocr_text))
                    myData.append(ocr_text)

                # cv2.imshow(y+str(x),imgCrop)
        #         cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
        #                     cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)

        #     with open('DataOutput.csv', 'a+') as f:
        #         for data in myData:
        #             f.write((str(data)+','))
        #         f.write('\n')

        #     #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
        #     print(myData)
        #     cv2.imshow(y+"2", imgShow)
        #     cv2.imwrite(y, imgShow)

        # # cv2.imshow("KeyPointsQuery",impKp1)
        # cv2.imshow("Output", imgQ)
        # cv2.waitKey(0)
        cv2.imshow('Output', imutils.resize(imgShow, width=700))
        #cv2.imshow('Output', imgQ)
        cv2.waitKey(0)


if __name__ == '__main__':
    Program = Form_detection()
    Program.main()
