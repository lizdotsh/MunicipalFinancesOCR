# input images into tabular form
import numpy as np 
import pandas as pd
import os
import sys
import logging
import re
import layoutparser as lp
from pyparsing import col
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Det2")
import yaml
with open('config.yml') as f:
    try:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print(dict)
    except yaml.YAMLError as e:
        print(e)
import ocr as o
import processing as pro


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

def save_det2_model(layout, pagenum = int, cfg=cfg):
    OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY']
    docname = cfg['SOURCE_NAME']
    if pagenum == None: 
        log.error('No pagenum specified')
    else: 
        df = layout.to_dataframe()
        df['pagenum'] = pagenum
        df['id'] = df.index
        log.info('Page {} converted to dataframe'.format(pagenum))
        if not os.path.exists('{}/{}/TableBank_model/'.format(OUTPUT_DIRECTORY,docname)): 
            os.makedirs('{}/{}/TableBank_model/'.format(OUTPUT_DIRECTORY, docname))
            log.warning("Directories not created for this project, created them.")
        if(os.path.isfile('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname))):
            log.info('csv exists, writing')
            df.to_csv('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname), mode='a', index= False,header=False)
            log.info('csv has been amended for page {} of {}'.format(pagenum, docname))
        else:
            log.warning('csv for {} does not exist, creating'.format(pagenum))
            df.to_csv('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname), index= False)
            log.info('csv created, model saved')


 #if __name__ == '__main__':
   # model = load_det2_model()
def already_in_csv(pagenum, cfg=cfg):
    """Tests if model layout has been amended to csv"""
    OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY']
    docname = cfg['SOURCE_NAME']
    if not os.path.exists('{}/{}/TableBank_model/'.format(OUTPUT_DIRECTORY,docname)): return False
    df = pd.read_csv('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname))
    if df['pagenum'].eq(pagenum).any(): 
        return True
    else: 
        return False
def load_det2_csv(pagenum, cfg=cfg): 
    """Loads det2 layout from csv"""
    OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY']
    docname = cfg['SOURCE_NAME']
    if(os.path.isfile('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname))):
        log.info('csv exists')
        df = pd.read_csv('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname))
        df = df[df['pagenum'] == pagenum]
        layout = lp.io.load_dataframe(df)
        return layout
    else: 
        log.error('CSV Doesnt exist')
   

def modeled_layout(image, pagenum = None, model = None, cfg = cfg, save = True):
    """
    Takes an image, and returns a Layout variable using the specified model. Then, it pads the model with specified padding. If model not specified, uses default model set by load_det2_model(). If padding not specified, uses padding set by config.yml. 

    Arguments: 
        image: ndarray, image to be used in model.detect()

        model: Detectron2 model. Will default to reloading the model if unspecified. 

        padding: dictionary of padding. needs to be in the form of a dictionary with 'left', 'right', 'top', and 'bottom' set to various integer values. Reccomended that you leave it default and set them with config.yml   
    """
    padding = cfg['Table']['Padding']
    OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY']
    docname = cfg['SOURCE_NAME']
    if already_in_csv(pagenum, cfg=cfg): 
        log.info('Already loaded this model, loading from CSV')
        return load_det2_csv(pagenum, cfg=cfg), model
    else: 
        if model == None:
            log.info("Model not specified, loading default from load_det2_model")
            model = load_det2_model()
        else: 
            log.info("Model Specified, continuing") 
        layout = model.detect(image).pad(**padding)
        log.info("Model ran on image")
        if save == True:
            if pagenum == None:
                log.error('No Pagenum Specified, cannot save')
                return layout, model
            else:  
                save_det2_model(layout, pagenum=pagenum, cfg=cfg)
                return layout, model
        else:
            return layout, model