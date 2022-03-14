import numpy as np 
import pandas as pd
import os
import sys
import logging
import re
import layoutparser as lp
from google.cloud import vision
from pyparsing import col
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("OCR")
import yaml
with open('config.yml') as f:
    try:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print(dict)
    except yaml.YAMLError as e:
        print(e)


def gcv_cred(keypath = cfg['Model']['GCV_KEY']):
    """
    Specifies Google Cloud Vision Credentials. Defaults to keypath in config.yml
    """
    x = lp.GCVAgent.with_credential(keypath,languages = ['en'])
    log.info("Specified GCV Credentials")
    return x
def gcv_res_exists(pagenum = None, cfg=cfg):
    """
    Checks to see if image has already been processed with google cloud
    """
    OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY']
    docname = cfg['SOURCE_NAME']
    if pagenum == None: return False 
    if(os.path.isfile('{}/{}/GCV_Res/{}-GCVRes.json'.format(OUTPUT_DIRECTORY, docname, pagenum))): return True 

def gcv_upload(image, ocr_agent = None): 
    """Sends image to GCV for detection. Typically used only by gcv_response, as gcv_response checks if this has already been done."""
    res = ocr_agent.detect(image, return_response = True)
    log.info('GCV processed/uploaded')
    return res, ocr_agent 

 
def gcv_response(image, pagenum = None, ocr_agent = None, cfg=cfg): 
    """
    This function takes an image and processes it on google cloud vision's OCR tool. If a docname and pagenum are specified, it saves them to /Output/docname/GCV_Res/pagenum-GCVres.json.

    Arguments: 
        image: ndarray for image to be processed

        pagenum: pagenum the image is from. Is used to name file when saving. 

        docname: Unique name for document. Use a different name for each, as it changes which folder it is saved in. 

        ocr_agent: The ocr_agent variable you set. If not loaded, will use gcv_cred() defaults and load it again. 
    """
    # Tests if working directory exists for specified docname.  
    OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY']
    docname = cfg['SOURCE_NAME']
    ocr = True
    if ocr_agent == None: 
        ocr = False
        log.warning('GCV credentials not loaded, loading with GCV_Cred()')
        ocr_agent = gcv_cred() 
    if pagenum == None:
        log.error("Pagenum not specified. Will not save response")
        res, ocr_agent = gcv_upload(image, ocr_agent=ocr_agent)
    else: 
        if bool(gcv_res_exists(pagenum=pagenum, cfg=cfg)): 
            log.info('GCV Already Processed {}, loading from file'.format(pagenum))
            res = ocr_agent.load_response('{}/{}/GCV_Res/{}-GCVRes.json'.format(OUTPUT_DIRECTORY, docname, pagenum))
        else:
            res, ocr_agent = gcv_upload(image, ocr_agent=ocr_agent) 
            if not os.path.exists('{}/{}/GCV_Res/'.format(OUTPUT_DIRECTORY,docname)): 
                os.makedirs('{}/{}/GCV_Res/'.format(OUTPUT_DIRECTORY, docname))
                log.warning("Directories not created for this project, created them.") 
            ocr_agent.save_response(res,'{}/{}/GCV_Res/{}-GCVRes.json'.format(OUTPUT_DIRECTORY,docname, str(pagenum)))
            log.info('GCV Saved')          
    return res, ocr_agent

def annotate_res(res, ocr_agent= None): 
    """
    Takes res file from GCV and splits it into layout files of three different aggregation levels. gcv_para is the paragraph level, gcv_word is the word level, and gcv_char is the character level. 

    Arguments: 
        res: The response from google cloud, see gcv_response() for more details
        ocr_agent: the ocr_agent, will load a new one if one is not specified with gcv_cred()
    
    Outputs: 
        gcv_block: Block level layout file
        gcv_para: Paragraph level layout file
        gcv_word: Word level layout file
        gcv_char: Character level layout file
        ocr_agent: Returns the ocr_agent variable if one was not specifed initially
    """
    ocr = True
    if ocr_agent == None: 
        ocr = False
        log.warning('OCR agent not specified. Loading with GCV_cred()')
        ocr_agent = gcv_cred()
        log.info('OCR agent loaded')
    else: log.info('OCR Agent Specified, continuing')
    gcv_block = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.BLOCK)
    log.info('Created gcv_block')
    gcv_para = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.PARA)
    log.info('Created gcv_para')
    gcv_word = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.WORD)
    log.info('Created gcv_word') 
    gcv_char = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.SYMBOL)
    log.info('Created gcv_word') 
    #if ocr == True: return gcv_block, gcv_para, gcv_word, gcv_char
    #else: return gcv_block, gcv_para, gcv_word, gcv_char, ocr_agent
    return gcv_block, gcv_para, gcv_word, gcv_char, ocr_agent
