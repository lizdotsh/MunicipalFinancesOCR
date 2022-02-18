# This file contains the functions used to process the preprocessed 
# input images into tabular form
import numpy as np 
import pdf2image
import pandas as pd
import os
import sys
import logging
import layoutparser as lp
from google.cloud import vision
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

def gcv_cred(keypath = cfg['Model']['GCV_KEY']):
    """
    Specifies Google Cloud Vision Credentials. Defaults to keypath in config.yml
    """
    x = lp.GCVAgent.with_credential(keypath,languages = ['en'])
    log.info("Specified GCV Credentials")
    return x

def gcv_response(image, pagenum = int, docname = str, ocr_agent = None): 
    """
    This function takes an image and processes it on google cloud vision's OCR tool. If a docname and pagenum are specified, it saves them to /Output/docname/GCV_Res/pagenum-GCVres.json.

    Arguments: 
        image: ndarray for image to be processed

        pagenum: pagenum the image is from. Is used to name file when saving. 

        docname: Unique name for document. Use a different name for each, as it changes which folder it is saved in. 

        ocr_agent: The ocr_agent variable you set. If not loaded, will use gcv_cred() defaults and load it again. 
    """
    # Tests if working directory exists for specified docname. 
    save = True
    ocr = True
    if docname == None: 
        save = False
        log.error("Docname not specified. Will not save response")
    if pagenum == None: 
        save = False
        log.error("Pagenumn not specified. Will not save response")

    if ocr_agent == None: 
        ocr = False
        log.warning('GCV credentials not loaded, loading with GCV_Cred()')
        ocr_agent = gcv_cred() 
    if not os.path.exists('Output/{}/GCV_Res/'.format(docname)): 
        os.makedirs('Output/{}/GCV_Res/'.format(docname))
        log.warning("Directories not created for this project, created them.")
    res = ocr_agent.detect(image, return_response = True)
    log.info('GCV processed, saving')
    if save == True:
        ocr_agent.save_response(res,'Output/{}/GCV_Res/{}-GCVRes.json'.format(docname, str(pagenum)))
        log.info('GCV Saved')
    else: log.info('Could not save GCV')
    if ocr == True: return res
    else: return res, ocr_agent
    

def annotate_res(res, ocr_agent= None):#-> gcv_word, gcv_para, gcv_char):
    """
    Takes res file from GCV and splits it into layout files of two different aggregation levels. gcv_word is the word level, and gcv_para is the paragraph level. 

    """
    ocr = True
    if ocr_agent == None: 
        ocr = False
        log.warning('OCR agent not specified. Loading with GCV_cred()')
        ocr_agent = gcv_cred()
        log.info('OCR agent loaded')
    else: log.info('OCR Agent Specified, continuing')

    gcv_para = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.PARA)
    log.info('Created gcv_para')
    gcv_word = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.WORD)
    log.info('Created gcv_word') 
    gcv_char = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.SYMBOL)
    log.info('Created gcv_word') 
    if ocr == True: return gcv_para, gcv_word, gcv_char
    if ocr == False: return gcv_para, gcv_word, gcv_char


        

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
    res, ocr_agent = gcv_response(image,1, 'test')
    gcv_para, gcv_word, gcv_char = annotate_res(res, ocr_agent)

    

