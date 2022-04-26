from cv2 import cv2
import numpy as np
import pytesseract
import os
import imutils
import re
import table_detection as td
import json

# C:\Program Files\Tesseract-OCR



"""
input invoice_config.json
output scan_data.json
"""

class Form_detection:
    def __init__(self):
        self.template = r'D:\Desktop\Desktop_2\1.Python_project\PDF_Automation\template\logistic_template.jpg'
        self.target = r''

        self.per = 25
        self.pixelThreshold = 500
        self.scale = 0.6
        self.roi = []
        self.myPoints = {}
        self.json_read()   # read invoice_config.json
        self.update_roi()  # update self.roi
        # self.roi = [[(1896, 764), (2348, 808), 'date', 'INVOICE DATE'],     
        #             [(1896, 840), (2340, 884), 'text', 'CUSTOMER ID'],
        #             [(1904, 916), (2340, 976), 'text', 'SHIPMENT'],
        #             [(1896, 995), (2340, 1050), 'date', 'DUE DATE'],
        #             [(1896, 1068), (2344, 1120), 'text', 'TERMS'],
        #             [(1885, 1125), (2344, 1184), 'text', 'CONSOL NUMBER'],
        #             [(148, 1592), (868, 1636), 'text', 'IMPORT CUSTOMS BROKER'],
        #             [(100, 1950), (2370, 2756), 'table', 'DESCRIPTION'],
        #             ]

        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        #config = r"-c tessedit_char_blacklist=—!$%«‘&\'*+-/;<=>?@[\\]^_`{|}~ --oem 3 --psm 3"
        self.config = r'-c tessedit_char_blacklist=«# '

    def json_read(self, jsonpath=r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\json\invoice_config.json'):
        """
        load data from json 
        ******load data at the begining******
        """
        if os.path.exists(jsonpath):
            f = open(jsonpath)
            data = json.load(f)  # data become dict
            self.myPoints = data['invoice_info_coordinate']
        else:
            print('[INFO] NO json.cannot load json')

    def update_roi(self):      # update roi lists
        self.roi = []
        for element in self.myPoints:
            image_info = [tuple(self.myPoints[element]['point1']), tuple(
                self.myPoints[element]['point2']), self.myPoints[element]['type'], self.myPoints[element]['name']]
            self.roi.append(image_info)


    def thresholding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 127,255 x   150,255 ok not best 170 not good
        img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)[1]
        #img=cv2.threshold(img, 70, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        return img

    def cleanup_text(self, text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 and ord(c) not in range(33, 40) and ord(c) not in range(42, 44) and ord(c) not in range(59, 64) else "" for c in text]).strip()

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

    # def json_createUpdate(self,jsonpath=r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\json\scan_data.json', content_dict={}):
    #     data ={}
    #     json_path = r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\json\scan_data.json'
    #     if os.path.exists(json_path):
    #         f = open(json_path)
    #         data = json.load(f)
    #     data.update(content_dict)
    #         ## Save our changes to JSON file
    #     jsonFile = open(json_path, "w+")
    #     jsonFile.write(json.dumps(data))
    #     jsonFile.close()

    def json_createUpdate(self, jsonpath=r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\json\scan_data.json', content_dict={}):
        # function to add to JSON
        data = {}
        if not os.path.exists(jsonpath):
            data['Form_detail'] = []
            with open(jsonpath, 'w') as outfile:
                json.dump(data, outfile)

        with open(jsonpath) as json_file:
            data = json.load(json_file)

            temp = data['Form_detail']
            # appending data to emp_details
            temp.append(content_dict)

        with open(jsonpath, 'w') as f:
            json.dump(data, f, indent=4)

    def align_jpg(self, img, imgQ):
        h, w, c = imgQ.shape
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        good = matches[:int(len(matches)*(self.per/100))]
        imgMatch = cv2.drawMatches(
            img, kp2, imgQ, kp1, good[:100], None, flags=2)

        srcPoints = np.float32(
            [kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32(
            [kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)

        return imgScan, imgShow, imgMask

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
            jpg_dict = {}
            in_file = os.path.join(path, y)
            self.head_tail = os.path.split(in_file)
            self.out_csv_path = os.path.join(
                r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\result", 'DataOutput-' + self.head_tail[1]+'.csv')

            jpg_dict = {'file': self.head_tail[1]}
            img = cv2.imread(path + "/"+y)

            # imgScan = ai.align_images(img,imgQ)
            # imgShow = imgScan.copy()
            # imgMask = np.zeros_like(imgShow)

            # not good
            # kp2, des2 = orb.detectAndCompute(img, None)
            # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            # matches = bf.match(des2, des1)
            # matches.sort(key=lambda x: x.distance)
            # good = matches[:int(len(matches)*(self.per/100))]
            # imgMatch = cv2.drawMatches(
            #     img, kp2, imgQ, kp1, good[:100], None, flags=2)

            # srcPoints = np.float32(
            #     [kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            # dstPoints = np.float32(
            #     [kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
            # imgScan = cv2.warpPerspective(img, M, (w, h))

            # imgShow = imgScan.copy()
            # imgMask = np.zeros_like(imgShow)

            imgScan, imgShow, imgMask = self.align_jpg(img, imgQ)

            myData = []

            print(
                f'################## Extracting Data from {self.head_tail[1]}  ##################')

            for x, r in enumerate(self.roi):
                cv2.rectangle(imgMask, (r[0][0], r[0][1]),
                              (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
                imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

                imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                hc, wc, cc = imgCrop.shape
                imgCrop = cv2.resize(
                    imgCrop, (int(wc*self.scale), int(hc*self.scale)))
                imgCrop_origin = imgCrop
                # imgCrop=self.thresholding(imgCrop)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(
                    imgCrop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                resizing = cv2.resize(border, None, fx=2,
                                      fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations=1)
                imgCrop = cv2.erode(dilation, kernel, iterations=2)

                if r[2] == 'text' or r[2] == 'date':
                    # imgCrop=thresholding(imgCrop)
                    ocr_text = pytesseract.image_to_string(
                        imgCrop)
                    ocr_text = self.filter_text(ocr_text)
                    if ocr_text == '':
                        print('[INFO] re-ocr the image')
                        ocr_text = pytesseract.image_to_string(
                            imgCrop_origin, config=self.config)
                        ocr_text = self.filter_text(ocr_text)

                    #ocr_text = self.filter_text(ocr_text)
                    print(
                        '{} :\n{} ,type={} \n**************'.format(r[3], repr(ocr_text), type(ocr_text)))
                    #cv2.imshow(r[3], imgCrop)
                    text_dict = {r[3]: ocr_text}
                    jpg_dict.update(text_dict)
                    myData.append(ocr_text)

                if r[2] == 'table':
                    pre_file = os.path.join(
                        r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\image", self.head_tail[1]+'-pre.jpg')
                    out_file = os.path.join(
                        r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\image", self.head_tail[1]+'-output.jpg')
                    table_csv_path = os.path.join(
                        r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\result", self.head_tail[1]+'table_result.csv')
                    df = td.table_detect_main(
                        img, r, pre_file, out_file, table_csv_path)
                    text_dict = {r[3]: df.to_dict()}
                    #cv2.imshow(r[3], imgCrop)
                    jpg_dict.update(text_dict)
                    myData.append(df.to_string())

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
                    #cv2.imshow(r[3], imgCrop)
                    text_dict = {r[3]: totalPixels}
                    jpg_dict.update(text_dict)
                    myData.append(totalPixels)

                if r[2] == '':
                    hc, wc, cc = imgCrop.shape
                    imgCrop = cv2.resize(
                        imgCrop, (int(wc*self.scale), int(hc*self.scale)))
                    imgCrop = self.thresholding(imgCrop)
                    ocr_text = pytesseract.image_to_string(
                        imgCrop, config=self.config)
                    ocr_text = self.filter_text(ocr_text)
                    print(
                        '{} :\n{}\n**************'.format(r[3], repr(ocr_text)))
                    #cv2.imshow(r[3], imgCrop)
                    text_dict = {r[3]: ocr_text}
                    jpg_dict.update(text_dict)
                    # myData.append(ocr_text)

                # cv2.imshow(y+str(x),imgCrop)
                cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
                            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)

            with open(self.out_csv_path, 'w') as f:
                for data in myData:
                    f.write((str(data)+','))
                f.write('\n')

            print('###############dict###############')

            print(jpg_dict)
            self.json_createUpdate(
                jsonpath=r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\json\scan_data.json', content_dict=jpg_dict)
            cv2.imshow('Output - '+self.head_tail[1], imutils.resize(imgShow, width=700))
        cv2.waitKey(0)


if __name__ == '__main__':
    Program = Form_detection()
    Program.main()
