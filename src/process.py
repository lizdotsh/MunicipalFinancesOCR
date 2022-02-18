# This file contains the functions used to process the preprocessed 
# input images into tabular form
import numpy as np 
import pdf2image
import pandas as pd
import sys
import logging
import layoutparser as lp
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Process")
import yaml
with open('/Users/liz/Documents/Projects/MunicipalFinancesOCR/config.yml') as f:
    try:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print(dict)
    except yaml.YAMLError as e:
        print(e)


def convert_PDF(file, pagenum):
    """
    Converts pdf to numpy array
        file: The file path of the PDF
        pagenum: the page number of the PDF to convert
    """
    pdf = np.asarray(pdf2image.convert_from_path(file)[pagenum])
    log.info("Converted page {} from {}".format(pagenum, file))   
    return pdf


def load_det2_model(DET_MODEL_PATH = cfg['Model']['DET_MODEL_PATH'], LABEL_MAP = cfg['Model']['LABEL_MAP']):
    """
    Loads the selected Delectron2 model
        DET_MODEL_PATH: Model path 
        LABEL_MAP: "Label Map" used by model, see LayoutParser website for more information 
        Defaults to values in config.yml
    """
    model = lp.Detectron2LayoutModel(
        config_path = DET_MODEL_PATH, 
        label_map = LABEL_MAP)
    log.info('Loaded Detectron2 model with LayoutParser')
    return model

def to_pos_id(y_1 = float, y_2 = float, pagenum = int, docheight = 3850) -> float:
    """
    Returns a position ID (float) of its position on the page as a decimal from 0 to 1, added on top of the pagenum

    Ex: A page that is 500px tall, with y_1 = 225 and y_2 = 275 on page 5. 
    Average height on page is 250px, so returns .5. Added to the page number gives the Position ID
    of 5.5. This is used to have a scalar value for the absolute position of objects in an entire document, instead of just simple pages. 
    Intended to be used to match up titles and tables that get cut off/on different pages from each other. 

    Arguments: 
        y_1: Lower Y value of the selection 
        y_2: Higher Y value of the selection 
        pagenum: Page number of the page where the selection came from 
        docheight: Height, in pixels, of the array of the document. Defaults to 3850, as that is dpi = 350 of the PDF
    """
    pos_id = float(pagenum) + (np.mean(y_1, y_2)/docheight)
    log.info("Created Log ID {} for page {}".format(pos_id, pagenum))
    return pos_id




#if __name__ == '__main__':
   # model = load_det2_model()

def modeled_layout(image, model = None, padding = cfg['Table']['Padding']):
    """
    Takes an image, and returns a Layout variable using the specified model. Then, it pads the model with specified padding. If model not specified, uses default model set by load_det2_model(). If padding not specified, uses padding set by config.yml. 

    Arguments: 
        image: ndarray, image to be used in model.detect()

        model: Detectron2 model. Will default to reloading the model if unspecified. 

        padding: dictionary of padding. needs to be in the form of a dictionary with 'left', 'right', 'top', and 'bottom' set to various integer values. Reccomended that you leave it default and set them with config.yml   
    """
    if model == None:
        log.info("Model not specified, loading default from load_det2_model")
        model = load_det2_model()
    else: log.info("Model Specified, continuing")
    try: 
        layout = model.detect(image).pad(**padding)
        log.info("Model ran on image")
        return layout
    except: 
        log.info("An error occured when trying to run the model")
           
def layout_excluding_layout(layout, filter_layout):
    """
    This function takes a Layout variable and removes all units that fall inside another Layout file
    Arguments: 
        layout: The source layout. In this case, the text layout. 
        filter_layout: The layout that the filter checks against. Everything from layout that lies within a unit of filter_layout will be removed by this function. 
        padding: automatic function to add padding 
    """
    x = lp.Layout([b for b in layout \
        if not any(b.is_in(b_tab) for b_tab in filter_layout)])
    log.info("Excluded filter_layout from the layout")
    return x
            
if __name__ == '__main__':
    image = np.asarray(pdf2image.convert_from_path('/Users/liz/Documents/Projects/LayoutParser/test.pdf')[1])
    modeled_layout(image)

    

