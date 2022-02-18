# This file contains the functions used to process the preprocessed 
# input images into tabular form
import numpy as np 
import pdf2image
import sys
import logging
import layoutparser as lp
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


def LoadDet2Model(DET_MODEL_PATH, LABEL_MAP):
    """
    Loads the selected Delectron2 model
        DET_MODEL_PATH: Model path 
        LABEL_MAP: "Label Map" used by model, see LayoutParser website for more information 
    """
    model = lp.Detectron2LayoutModel(
        config_path = DET_MODEL_PATH, 
        label_map = LABEL_MAP)
    return model