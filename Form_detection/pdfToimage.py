import os
from pdf2image import convert_from_path




def _pdf2image(output_dir,pdf_file):
    if pdf_file.endswith(".pdf"):
        pages = convert_from_path(pdf_file, 300,poppler_path=r'C:\Program Files\poppler-0.68.0\bin')
        #pages = convert_from_path(pdf_file, 300)
        print(pdf_file)
        pdf_file = pdf_file[:-4]

        for page in pages:
            path = os.path.join(output_dir,"%s-page-%d.jpg" % (pdf_file,pages.index(page)+1))
            #print(path,"JPEG")
            page.save(path)

if __name__ == '__main__':
    pdf_dir = r"D:\Desktop\Desktop_2\1.Python_project\PDF_Automation\source"
    img_dir = r"D:\Desktop\Desktop_2\1.Python_project\PDF_Automation\image"
    print('pdftoimage main')
    os.chdir(pdf_dir)
    for pdf_file in os.listdir(pdf_dir):
        _pdf2image(img_dir,pdf_file)
