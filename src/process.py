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
from pyparsing import col
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Process")
import yaml
with open('config.yml') as f:
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
    pdf = np.asarray(pdf2image.convert_from_path(file, dpi=250, first_page=pagenum, last_page=pagenum)[0])
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
    if not os.path.exists('{}/{}/GCV_Res/'.format(cfg['OUTPUT_DIRECTORY'],docname)): 
        os.makedirs('{}/{}/GCV_Res/'.format(cfg['OUTPUT_DIRECTORY'], docname))
        log.warning("Directories not created for this project, created them.")
    res = ocr_agent.detect(image, return_response = True)
    log.info('GCV processed, saving')
    if save == True:
        ocr_agent.save_response(res,'{}/{}/GCV_Res/{}-GCVRes.json'.format(cfg['OUTPUT_DIRECTORY'],docname, str(pagenum)))
        log.info('GCV Saved')
    else: log.info('Could not save GCV')
    if ocr == True: return res
    else: return res, ocr_agent
    

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
    if ocr == True: return gcv_block, gcv_para, gcv_word, gcv_char
    else: return gcv_block, gcv_para, gcv_word, gcv_char, ocr_agent


        

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

def save_det2_model(layout, pagenum = int, OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY'], docname = cfg['SOURCE_NAME']):
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
def already_in_csv(pagenum, OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY'], docname = cfg['SOURCE_NAME']):
    """Tests if model layout has been amended to csv"""
    if not os.path.exists('{}/{}/TableBank_model/'.format(OUTPUT_DIRECTORY,docname)): return False
    df = pd.read_csv('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname))
    if df['pagenum'].eq(pagenum).any(): 
        return True
    else: 
        return False
def load_det2_csv(pagenum, OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY'], docname = cfg['SOURCE_NAME']): 
    """Loads det2 layout from csv"""
    if(os.path.isfile('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname))):
        log.info('csv exists')
        df = pd.read_csv('{}/{}/TableBank_model/{}.csv'.format(OUTPUT_DIRECTORY, docname, docname))
        df = df[df['pagenum'] == pagenum]
        layout = lp.io.load_dataframe(df)
        return layout
    else: 
        log.error('CSV Doesnt exist')
    

def modeled_layout(image, pagenum = None, model = None, padding = cfg['Table']['Padding'], save = True, OUTPUT_DIRECTORY = cfg['OUTPUT_DIRECTORY'], docname = cfg['SOURCE_NAME']):
    """
    Takes an image, and returns a Layout variable using the specified model. Then, it pads the model with specified padding. If model not specified, uses default model set by load_det2_model(). If padding not specified, uses padding set by config.yml. 

    Arguments: 
        image: ndarray, image to be used in model.detect()

        model: Detectron2 model. Will default to reloading the model if unspecified. 

        padding: dictionary of padding. needs to be in the form of a dictionary with 'left', 'right', 'top', and 'bottom' set to various integer values. Reccomended that you leave it default and set them with config.yml   
    """

    if already_in_csv(pagenum, OUTPUT_DIRECTORY = OUTPUT_DIRECTORY, docname=docname): 
        log.info('Already loaded this model, loading from CSV')
        return load_det2_csv(pagenum, OUTPUT_DIRECTORY=OUTPUT_DIRECTORY, docname = docname)
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
                return layout
            else:  
                save_det2_model(layout, pagenum=pagenum, OUTPUT_DIRECTORY=OUTPUT_DIRECTORY, docname=docname)
                return layout
        else:
            return layout
   
           
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


def text_layout_from_selection(text_layer, bounding_layer):
    """Returns text from selected boundary layer"""
    return text_layer.filter_by(bounding_layer)

def list_text_layout_from_selection(text_layer, bounding_layers):
    """Inputs a text layer a Layer of several bounding layers. It returns the text from each individual bounding polygon using text_from_selection() and returns them in list form."""
    l = []
    for i,x in enumerate(bounding_layers): 
        z = text_layout_from_selection(text_layer, x)
        log.info('Converted bounding layer {}'.format(i))
    return z

def remove_titles(bounding_layers, cfgtable = cfg['Table']):
    px = cfgtable['titlerow_px'] + cfgtable['Padding']['top']
    return bounding_layers.pad(top=-px)

def to_polygons(bounding_layer):
    """Converts Layer object into a simpler polygon"""
    return bounding_layer.get_homogeneous_blocks()

def create_bounding_polygons(bounding_poly = lp.elements.layout_elements.TextBlock, column_dict = cfg['Table']):
    """
    Creates a list of bounding polygons for a table given a polygon for the table. Specifications need to be given in config.yml

    Arguments: 
        bounding_poly: Must be a SINGULAR text block rectangle. This means if you just passed a layer file through to_polygons(), you must feed the function a specific polyon in that list. Ex: table_poly[0]
        column_dict: You can specify a different dictionary specifying the columns if you want. Check config.yml for template. 
    """
    bounding_coords = bounding_poly.coordinates
    bounding_width = bounding_poly.width
    col_widths_frac = []
    for i in column_dict["columns"]:
        col_widths_frac.append(column_dict["columns"][i]["width"]) # Gather a list of all the widths in config.yml
    col_widths_px = []
    for x in col_widths_frac:
        col_widths_px.append(x * bounding_width) # Convert those widths into pixel values based on bounding_layer 
    current_pos = bounding_coords[0]
    polygons = []
    for _ in col_widths_px:
        x = current_pos + _
        polygons.append(
            lp.Rectangle(
                x_1 = current_pos,
                y_1 = bounding_coords[1],
                x_2 = x,
                y_2 = bounding_coords[3],
            )
        )
        current_pos = x
    return polygons


def main():
    image = np.asarray(pdf2image.convert_from_path('/Users/liz/Documents/Projects/LayoutParser/test.pdf')[1])
    table_layout = modeled_layout(image)
    res, ocr_agent = gcv_response(image,1, 'test')
    gcv_block, gcv_para, gcv_word, gcv_char = annotate_res(res, ocr_agent)
    table_poly = to_polygons(table_layout)
    #testing create polygons
    ll = create_bounding_polygons(remove_titles(table_poly[1]))
    hi = gcv_word.filter_by(ll[0], soft_margin = {"left":10, "right":10})
    lp.draw_box(image, ll, box_width=4).save("Tests/bruh3.png", "PNG")
    lp.draw_box(image, hi, box_width=4).save("Tests/bruh4.png", "PNG") 
if __name__ == '__main__':
    log.info('for debug only')
    #main()

   

