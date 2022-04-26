import os
import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
#import imutils  #for testing main
# This only works if there's only one table on a page
# Important parameters:
#  - morph_size
#  - min_text_height_limit
#  - max_text_height_limit
#  - cell_threshold
#  - min_columns

def cleanup_text(text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
    return "".join([c if ord(c) < 128 and ord(c) not in range(33,40) and ord(c) not in range(42,44) and ord(c) not in range(59,64) else "" for c in text]).strip()


def filter_text(ocr_text):
    ocr_text = cleanup_text(ocr_text)
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



def pre_process_image(img, save_in_file, morph_size=(30, 5)):   #best(30 ,5)    default (8,8)

    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    #pre = cv2.threshold(pre, 70, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    pre = cv2.threshold(pre, 70, 255, cv2.THRESH_BINARY)[1]    #70,255

    cv2.imshow(save_in_file+'thre',pre)
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    cv2.imshow(save_in_file+'dil',cpy)
    pre = ~cpy
    

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre



def find_text_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4


    # contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # # Getting the texts bounding boxes based on the text size assumptions
    # boxes = []
    # for contour in contours:
    #     box = cv2.boundingRect(contour)
    #     h = box[3]

    #     if min_text_height_limit < h < max_text_height_limit:
    #         boxes.append(box)
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x:x[1][1]))
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w<1000 and h<500):
            #image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            boxes.append([x,y,w,h])

    return boxes,boundingBoxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # Filtering out the clusters having less than 2 cols
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    return table_cells


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    return hor_lines, ver_lines

def scale_tur_list(boxes,scale):
    temp_list =[]
    for k in boxes:
        temp_list.append(list(k))
    for rol,tur in enumerate(temp_list):
        for col,item in enumerate(tur):
            temp_list[rol][col] = int(temp_list[rol][col]*scale)
    boxes=[]
    for i in temp_list:
        boxes.append(tuple(i))

    return boxes

def get_table_detail(boxes,boundingBoxes):
    rows=[]
    columns=[]
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    #print(mean)
    columns.append(boxes[0])
    previous=boxes[0]
    for i in range(1,len(boxes)):
        if(boxes[i][1]<=previous[1]+mean/2):
            columns.append(boxes[i])
            previous=boxes[i]
            if(i==len(boxes)-1):
                rows.append(columns)
        else:
            rows.append(columns)
            columns=[]
            previous = boxes[i]
            columns.append(boxes[i])
    # print("Rows")

    for row in rows:
        #print(row)
        total_cells=0
        for i in range(len(row)):
            if len(row[i]) > total_cells:
                total_cells = len(row[i])
        #print(total_cells)
        center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
        #print(center)
        center=np.array(center)
        center.sort()
        #print(center)

    return rows,total_cells,center


def box_coord_listCreate(rows,total_cells,center):
    boxes_list = []
    for i in range(len(rows)):
        l=[]
        for k in range(total_cells):
            l.append([])
        for j in range(len(rows[i])):
            diff = abs(center-(rows[i][j][0]+rows[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            l[indexing].append(rows[i][j])
        boxes_list.append(l)
    # for box in boxes_list:
    #     print(box)
    return boxes_list

def ocr_box(boxes_list,img):
    dataframe_final=[]
    print('total row=',len(boxes_list))
    print()
    for i in range(len(boxes_list)):     # i = row
        print('total colium=',len(boxes_list[i]))
        for j in range(len(boxes_list[i])):  # j = column
            if(len(boxes_list[i][j])==0):     #fill none if no object detected
                print('[INFO] add space')
                dataframe_final.append(None)
            else:
                for k in range(len(boxes_list[i][j])):
                    print('i={},j={},k{}'.format(i,j,k))
                    y,x,w,h = boxes_list[i][j][k][0],boxes_list[i][j][k][1], boxes_list[i][j][k][2],boxes_list[i][j][k][3]
                    roi = img[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(roi,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=2)   
                    # cv2.imshow('erosion-'+str(i)+','+str(j), erosion)    
                    # cv2.waitKey(0)         
                    out = pytesseract.image_to_string(erosion)
                    out = filter_text(out)
                    print('out=',repr(out))

                    # if(len(out)==0):  # no string
                    #     hc, wc, cc = img.shape
                    #     scale = 0.5
                    #     erosion_new= cv2.resize(erosion, (int(wc*scale), int(hc*scale)))
                    #     print('[INFO] re-ocr the image table')
                    #     out = pytesseract.image_to_string(erosion_new)
                    #     out = filter_text(out)
                    #     cv2.imshow('re ocr origin-'+str(i)+','+str(j), erosion_new) # show the origin
                    #     print('re-out=',repr(out))
                    #s = s +" "+ out
                #print('out=',repr(out))
                if repr(out) != '':
                    dataframe_final.append(out) 
                print(dataframe_final)
                

    print(dataframe_final)
    print('arr = ',dataframe_final)
    return dataframe_final

def map_table(dataframe_final,rows,total_cells):
    arr = np.array(dataframe_final)

    dataframe = pd.DataFrame(arr.reshape(len(rows), total_cells))

    for column in dataframe:   # check empty, delete empty column
        if dataframe[column].isnull().sum() == len(dataframe.index):
            del dataframe[column]

    # data = dataframe.style.set_properties(align="left")
    # #print(data)
    # #print(dataframe)
    # d=[]
    # for i in range(0,len(rows)):
    #     for j in range(0,total_cells):
    #         print(dataframe[i][j],end=" ")
    # print()
    return dataframe

def scan_table(text_boxes,boundingBoxes,img):
    #ocr text in table
    rows,total_cells,center = get_table_detail(text_boxes,boundingBoxes)
    boxes_list=box_coord_listCreate(rows,total_cells,center)
    dataframe_final=ocr_box(boxes_list,img)
    df = map_table(dataframe_final,rows,total_cells)

    print('#########################Table Dataframe#########################')
    #df=df.str.replace(r'\n','')
    df = df.apply(lambda col: col.str.replace(r'\n', r''))
    print(df)
    print('#################################################################')
    return df
    
def table_detect_main(img=None,roi=None,pre_file=r'',out_file=r'',out_csv_path=r'',scale=0.5):  # real main
    img = img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    hc, wc, cc = img.shape
    img= cv2.resize(img, (int(wc*scale), int(hc*scale)))
    print('[INFO] process image...')
    pre_processed = pre_process_image(img, pre_file)
    
    print('[INFO] find text box...')
    text_boxes,boundingBoxes = find_text_boxes(pre_processed) 
    
    print('[INFO] visualize table grid')
    # Visualize the result
    cells = find_table_in_boxes(text_boxes)
    hor_lines, ver_lines = build_lines(cells)
    vis = img.copy()
    for line in hor_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for line in ver_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite(out_file, vis)
    cv2.imshow('table-'+pre_file, vis)  # show the table
    

    #cv2.waitKey(0)
    #ocr text in table
    print('[INFO] start scanning image by OCR')
    df = scan_table(text_boxes,boundingBoxes,img)
    print('[INFO] convert table data to csv')
    df.to_csv(out_csv_path, index=False)
    print('[INFO] csv is saved in '+out_csv_path)
    return df


def  test_main():  # for local testing 
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    original_dir = r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\UserForms"
    for count,img_file in enumerate(filter(lambda f: (f.lower().endswith('.jpg')==True), os.listdir(original_dir))):
        in_file = os.path.join(r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\UserForms", img_file)
        #in_file = os.path.join(r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\UserForms", r'OCB1-page-1.jpg')
        head_tail = os.path.split(in_file)
        pre_file = os.path.join(r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\image", head_tail[1]+'-pre.jpg')
        out_file = os.path.join(r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\image", head_tail[1]+'-output.jpg')
        out_csv_path = os.path.join(r"D:\Desktop\Desktop_2\1.Python_project\Form_detection\result", head_tail[1]+'table_result.csv')
        roi = [(100, 1954), (2370, 2756), 'table', 'DESCRIPTION'] # OEH1-page-1.jpg
        #roi = [(100, 1900), (2370, 2756), 'table', 'DESCRIPTION'] # OEH1-page-1.jpg    #include header DESCRIPTION  & CHARGES IN HKD.
        if(img_file==r'20210126121312567204-page-1.jpg'):
            roi = [(244, 324), (2240, 524), 'table', 'DETAIL']  # 20210126121312567204-page-1.jpg
            continue
        scale=0.5
        
        print('[INFO] start handling='+head_tail[1]+'...')
        
        img = cv2.imread(os.path.join(in_file))

        # img = img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
        # hc, wc, cc = img.shape
        # img= cv2.resize(img, (int(wc*scale), int(hc*scale)))
        table_detect_main(img,roi,pre_file,out_file,out_csv_path)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_main()
