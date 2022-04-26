######### Region Selector ###############################
"""
This script allows to collect raw points from an image.
The inputs are two mouse clicks one in the x,y position and
the second in w,h of a rectangle.
Once a rectangle is selected the user is asked to enter the type
and the Name:
Type can be 'Text' or 'CheckBox'
Name can be anything


To close the window, press "S"
"""

from cv2 import cv2
import random
import os
import json


class RegionSelector:
    def __init__(self):
        self.path = r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\UserForms\OEH1-page-1.jpg'
        self.scale = 0.25
        self.circles = []
        self.counter = 0
        self.counter2 = 0
        self.point1 = []
        self.point2 = []
        self.myPoints = {}
        self.myColor = []
        self.content_dict = {}
        self.json_size = 0
        self.roi = []   # particiular image from png
        self.img = None

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


    def json_createUpdate_list(self, jsonpath=r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\json\invoice_config.json', content_dict={}):
        print("[INFO] print json")
        data = {}
        data['invoice_info_coordinate'] = content_dict    # create json file
        with open(jsonpath, 'w') as outfile:
            json.dump(data, outfile)

    def imgDisplayUpdate(self):
        # show the rect image
        for index, r in enumerate(self.roi):
            cv2.rectangle(self.img, (r[0][0], r[0][1]),
                (r[1][0], r[1][1]), (0, 255, 0),3)


    def mousePoints(self, event, x, y, flags, params):
        '''
        # press S key to terminate layout
        '''
        image_roi = {}
        typeNum = 0
        # global self.counter,self.point1,self.point2,self.counter2,self.circles,self.myColor
        if event == cv2.EVENT_LBUTTONDOWN:

            if self.counter == 0:
                self.point1 = int(x//self.scale), int(y//self.scale)
                self.counter += 1
                self.myColor = (random.randint(
                    0, 2)*200, random.randint(0, 2)*200, random.randint(0, 2)*200)
            elif self.counter == 1:
                self.point2 = int(x//self.scale), int(y//self.scale)
                cv2.imshow("Original Image ", self.img)
                name = input('[INPUT] Enter Field Name: ')
                while(True):
                    typeNum = input(
                        '[MENU] Select Field Type: \n\t1:text \n\t2:date \n\t3:table \n\t4.number only \nEnter number:')
                    try:
                        typeNum = int(typeNum)  # string to int
                        if not (1 <= typeNum <= 4):
                            raise Exception
                        else:
                            break  # input success
                    except Exception as e:
                        print('[WARNING] Invalid input')
                        pass

                if typeNum == 1:
                    type = 'text'
                elif typeNum == 2:
                    type = 'date'
                elif typeNum == 3:
                    type = 'table'
                elif typeNum == 4:
                    type = 'numOnly'

                print('[INFO] Point out next area')
                #self.myPoints.append([self.point1, self.point2, type, name])
                image_roi['name'] = name
                image_roi['type'] = type
                image_roi['point1'] = self.point1
                image_roi['point2'] = self.point2
                self.myPoints[len(self.myPoints)] = image_roi
                self.update_roi()
                self.counter = 0

            self.circles.append([x, y, self.myColor])
            
            self.counter2 += 1

            
    def main(self):

        self.img = cv2.imread(self.path)
        
        self.json_read()
        self.update_roi()
        self.imgDisplayUpdate()
        self.img = cv2.resize(self.img, (0, 0), None, self.scale, self.scale)
        while True:
            # To Display points
            index = 0
            for index in range(len(self.circles)):
                #cv2.circle(self.img, (self.circles[index][0], self.circles[index][1]), 3, self.circles[index][2], cv2.FILLED)
                if index % 2 == 0 and (len(self.circles) > index+1):
                    cv2.rectangle(self.img, (self.circles[index][0], self.circles[index][1]),(self.circles[index+1][0], self.circles[index+1][1]), (255, 0, 0),2)
                    

            for index,element in enumerate(self.circles):
               cv2.circle(self.img, (element[0], element[1]), 3, element[2], cv2.FILLED)  # element[0] = x ,element[1] = y ,element[2] = color
            
            
            cv2.imshow("Original Image ", self.img)
            cv2.setMouseCallback("Original Image ", self.mousePoints)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.json_createUpdate_list(
                    jsonpath=r'D:\Desktop\Desktop_2\1.Python_project\Form_detection\json\invoice_config.json', content_dict=self.myPoints)

                break


if __name__ == '__main__':
    Program = RegionSelector()
    Program.main()
