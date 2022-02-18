# This file contains the functions used to process the preprocessed 
# input images into tabular form
import numpy as np 
import pdf2image
import sys
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Process")

def ConvertPDF(file, pagenum):
    """
    Converts pdf to numpy array
        file: The file path of the PDF
        pagenum: the page number of the PDF to convert
    """
    pdf = np.asarray(pdf2image.convert_from_path(file)[pagenum])
    log.info("Converted page {} from {}".format(pagenum, file))   
    return pdf


