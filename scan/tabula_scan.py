import tabula  # java is a must
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from PyPDF2 import PdfFileReader
from PyPDF2.generic import RectangleObject
file = r"D:\Git Project\firebase-project\source\8526429315 202202.pdf"
page = 4  # page 1 -7
# PDF file to extract tables from
# tabula


def tabula_scan(left_top_tuple, right_top_tuple, bottom_tuple, page_tuple):
    print("left top="+str(left_top_tuple))
    print("right top="+str(right_top_tuple))
    print("bottom ="+str(bottom_tuple))
    print("full page size="+str(page_tuple))

    # testing area
    # top=350
    # height=380
    # bottom=top+height
    # left=37
    # width=600
    # right=left+width

    # target area
    top = page_tuple[1] - left_top_tuple[1] + 10
    # print("top = %s" % top)
    height = left_top_tuple[1] - bottom_tuple[1] - 10
    # print("left top y= %s   bottom top y= %s height=%s" %
    #       (str(left_top_tuple[1]), str(bottom_tuple[1]), str(height)))
    bottom = top+height
    left = left_top_tuple[0]
    width = right_top_tuple[0] - left_top_tuple[0] + 90
    # print("left top x= %s   right top x= %s width=%s" %
    #       (str(left_top_tuple[0]), str(right_top_tuple[0]), str(width)))
    right = left+width

    tabula.util.java_version()
    dfs = tabula.read_pdf(file, pages=page, pandas_options={'header': None}, encoding='big5', guess=False, stream=True, multiple_tables=False, area=[top, left, bottom, right],
                          columns=(50, 90, 165, 220, 250, 315, 340, 400, 450, 500, 550, 600))

    dfs[0] = dfs[0].set_axis(['序號', '日期', '單號', '寄件', '地區', '收件',
                             '地區.1', '電話', '費用類別', '重量', '原運費', '折扣後'], axis=1, inplace=False)
    print(dfs[0])
    # print(dfs[0]['序號'])
    # print(dfs[0]['日期'])
    # print(dfs[0]['單號'])
    # print(dfs[0]['寄件'])
    # print(dfs[0]['地區'])
    # print(dfs[0]['收件'])
    # print(dfs[0]['地區.1'])
    # print(dfs[0]['電話'])
    # print(dfs[0]['費用類別'])
    # print(dfs[0]['重量'])
    # print(dfs[0]['原運費'])
    # print(dfs[0]['折扣後'])


def get_table_area(target_page_num=0, keyword=r'序號') -> tuple:
    fp = open(file, 'rb')
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(fp)

    left_top_bool = False
    right_top_bool = False
    bottom_bool = False
    last_page_bool = False
    left_top_tuple = ()
    right_top_tuple = ()
    bottom_tuple = None
    last_page_num =0

    for page_num, page in enumerate(pages):
        print('Processing next page...')
        print("processing page number = %s" % page_num)
        interpreter.process_page(page)
        layout = device.get_result()
        for lobj in layout:
            if isinstance(lobj, LTTextBox):
                x, y, text = lobj.bbox[0], lobj.bbox[3], lobj.get_text()
                if keyword in text and not left_top_bool and page_num == target_page_num-1:
                    print('left top At %r is text: %s' % ((x, y), repr(text)))
                    left_top_tuple = (x, y)
                    left_top_bool = True

                if r'折扣後' in text and not right_top_bool and page_num == target_page_num-1:
                    print('right top At %r is text: %s' % ((x, y), repr(text)))
                    right_top_tuple = (x, y)
                    right_top_bool = True


                if r'合計' in text:
                    last_page_num = page_num +1
                    print('last page At  page %r is text: %s' % (last_page_num, repr(text)))
                    if not bottom_bool:
                        bottom_tuple = (x, y)
                        bottom_bool = True
                    
                    return (left_top_tuple, right_top_tuple, bottom_tuple)
                    
                if r'第' in text and r'頁' in text and not bottom_bool and page_num == target_page_num-1:
                    print('bottom At %r is text: %s' % ((x, y), repr(text)))
                    bottom_tuple = (x, y)
                    bottom_bool = True

                # if left_top_bool and bottom_bool and right_top_bool:
                #     print("get area success")
                #     break
    print("Not success")               
    return None,None,None

def get_page_size(target_page_num=0) -> tuple:
    input1 = PdfFileReader(open(file, 'rb'))
    print('PDF page size =')
    print("page %s" % page)
    print(input1.getPage(target_page_num).mediaBox)
    x, y = input1.getPage(0).mediaBox.upperRight
    return (x, y)


if __name__ == '__main__':
    # tabula_scan()
    page_tuple = get_page_size(page)
    left_top_tuple, right_top_tuple, bottom_tuple = get_table_area(
        target_page_num=page)
    tabula_scan(left_top_tuple, right_top_tuple, bottom_tuple, page_tuple)
