# This file contains the functions used to process the preprocessed 
# input images into tabular form
import numpy as np 
import pdf2image
import pandas as pd
import os
import sys
import logging
import re
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
        if gcv_res_exists(pagenum=pagenum, cfg=cfg): 
            log.info('GCV Already Processed, loading from file')
            res = ocr_agent.load_response('{}/{}/GCV_Res/{}-GCVRes.json'.format(OUTPUT_DIRECTORY, docname, pagenum))
        else:
            res, ocr_agent = gcv_upload(image, ocr_agent=ocr_agent) 
            if not os.path.exists('{}/{}/GCV_Res/'.format(OUTPUT_DIRECTORY,docname)): 
                os.makedirs('{}/{}/GCV_Res/'.format(OUTPUT_DIRECTORY, docname))
                log.warning("Directories not created for this project, created them.") 
            ocr_agent.save_response(res,'{}/{}/GCV_Res/{}-GCVRes.json'.format(OUTPUT_DIRECTORY,docname, str(pagenum)))
            log.info('GCV Saved')          
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
    pos_id = float(pagenum) + (((y_1 + y_2)/2)/docheight)
    log.info("Created Log ID {} for page {}".format(pos_id, pagenum))
    return pos_id

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
        return load_det2_csv(pagenum, cfg=cfg)
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
                save_det2_model(layout, pagenum=pagenum, cfg=cfg)
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
    log.info("excluded filter_layout from the layout")
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

def remove_many_titles(table_poly):
    l = []
    for i in table_poly:
        _ = remove_titles(i)
        l.append(_)
    return l

 
def isolate_titles(bounding_poly, cfgtable = cfg['Table']):
    """Takes padded bounding polygon of table, removes padding, and then removes all put the pixels set in titlerow_px to isolate the title row"""
    no_toptitle = bounding_poly.pad(top = -cfgtable['Padding']['top'])
    height = no_toptitle.height
    px = height - cfgtable['titlerow_px'] 
    isolated = no_toptitle.pad(bottom=-px)
    log.info('isolated title')
    return isolated

def to_polygons(bounding_layer):
    """Converts Layer object into a simpler polygon"""
    return bounding_layer.get_homogeneous_blocks()

def cols_px(bounding, cfgtable = cfg['Table']):
    """
    uses settings in config.yml to determine the location of each column based on the location of the column titles. 
    
    """
    df = bounding.to_dataframe() # Turns into dataframe
    df['x_1'] = [i[0] for i in df['points']] # separates coordinates 
    df['x_2'] = [i[2] for i in df['points']]
    df['x_avg'] = (df['x_1'] + df['x_2'])/2
    namedf = pd.DataFrame(columns=['ColNum', 'x_avg'])
    for x,i in enumerate(cfgtable['columns']):
        for _ in df['text']:
            if bool(re.search(str(cfgtable['columns'][i]['regex']), _.lower())): 
                df1 = df[df['text'] == _][['x_avg']]
                df1['ColNum'] = x
                namedf = namedf.append(df1)
                log.info('Appended column {}'.format(x))
       # re.search([i]['regex'])
    return namedf.reset_index(drop=True)

def column_poly(table_text_poly, cols_px_df, gcv_word, cfgtable = cfg['Table']):
    """
    Takes column positions in the form of a dataframe from cols_px() returns polygons for each section. 
        
        Arguments: 
        table_text_poly: The source polygon. if a layerfile, make sure to set [i] on the end to get a specific table. 
        cols_px_df: df given by cols_px
        cfgtable: config file's table section, will work automatically by default
    Other important information: 

        - Each column will have a "special" section. at the moment you should only really have that set on the first column, as I am using it for dynamic sizing for
        the first column only and it is kinda hardcoded at the moment. special gap sets the gap. check the comments on the code of the function for more information. 
    """
    coords = table_text_poly.coordinates # gets coordinates of the bounding layer
    layouts = []
    
    for i,x in enumerate(cfgtable['columns']):
             rec = lp.Rectangle(
                x_1 = (cols_px_df['x_avg'][i] - cfgtable['columns'][x]['hard_margin']['left']),
                y_1 = coords[1],
                x_2 = (cols_px_df['x_avg'][i] + cfgtable['columns'][x]['hard_margin']['right']),
                y_2 = coords[3])
             layouts.append(rec)     
             if cfgtable['columns'][x]['special'] == True: 
                 spl = x
    tb = layouts[0].coordinates # checks coordinates of title_of_bond and stores them 
    ri = layouts[1].coordinates # checks coordinates for the second row, as it will use the second row to dynamically assign the first row
    layouts[0] = lp.Rectangle(
        # This whole section just sets the first row's x_2 to the x_1 of the second row, minus a special_gap
        x_1 = tb[0],
        y_1 = tb[1],
        x_2 = ( ri[0] - cfgtable['columns'][spl]['special_gap']),
        y_2 = tb[3]
    )
    return layouts

def txt_from_col_poly(col_poly, gcv_word, cfgtable = cfg['Table']): 
    """
    Takes the output from identify_rows and puts everything in a nice data frame. 
    """
    cols = []
    for i,x in enumerate(cfgtable['columns']):
        filtered = gcv_word.filter_by(
            col_poly[i], 
            soft_margin = cfgtable['columns'][x]['soft_margin']
        , center = True)
        cols.append(filtered)
    return cols


def identify_rows(col_txt_list, distance_th, gcv_word, cfgtable = cfg['Table']): 
    """
    used to dynamically calculate rows based on distances between the x values of various things. 
    col_text_list must be a list of polygons defining the various different columns, and must not have a title in it. 
    distance_th is the distance between the center of the polygons 
    gcv_word is google cloud word thing 
    returns list of list, first subsetting is by col second is by row. 
    """
    list = []
    #df = pd.DataFrame()
    blocks = txt_from_col_poly(col_txt_list, gcv_word, cfgtable)
    o = 0
    for i in blocks: 
        i = sorted(i, key = lambda x: x.coordinates[1]) # Sort the blocks vertically from top to bottom
        distances = np.array([((b2.coordinates[1] + b2.coordinates[3])/2) - ((b1.coordinates[3] + b1.coordinates[1])/2) for (b1, b2) in zip(i, i[1:])])
        # Calculate the distances:
        # y coord for the upper edge of the bottom block -
        #   y coord for the bottom edge of the upper block
        # And convert to np array for easier post processing
        distances = np.append([0], distances) # Append a placeholder for the first word
        block_group = (distances>distance_th).cumsum() # Create a block_group based on the distance threshold
        grouped_blocks = [lp.Layout([]) for i in range(max(block_group)+1)]
        for _, block in zip(block_group, i):
            grouped_blocks[_].append(block)   
    #    df[str(i)] = pd.Series(grouped_blocks) 
        list.append(grouped_blocks)
       
    return list

def layer_to_df(double_layered_list): 
    df = pd.DataFrame()
    for p,i in enumerate(double_layered_list): 
        list = []
        for u in i: 
            textlist = u.get_texts()
            text = ' '.join(str(e) for e in textlist)
            list.append(text)
        df[str(p)] = pd.Series(list)
    return df
  
def cont_or_not(table_poly, gcv_word, cfgtable=cfg['Table']):
    title = table_poly.pad(-cfgtable['Padding']['top'])
    titletext = gcv_word.filter_by(
       title, 
       soft_margin = cfgtable['cont']['cont_soft_margin']
    )
    texts = titletext.get_texts()
    text = ' '.join(str(e) for e in texts)
    if bool(re.search(str(cfgtable['cont']['regex']), text.lower())): 
        return True
    else: return False


    
def parse_table(table_layout, gcv_word, tablenum, cfgtable = cfg['Table']): 
    """
    combines many functions into one. Essentially all you need is the table layout, gcv_word, 
    and tablenum and it will return you a dataframe of all of the text within each table
    """
    table_poly = to_polygons(table_layout) # Convert to polygon 
    tabletitletext = gcv_word.filter_by(
       isolate_titles(table_poly[tablenum], cfgtable),
       soft_margin = cfgtable['title_soft_margin'] 
    )
    px = cols_px(tabletitletext) # location of each column
    if px.isnull().values.any(): 
            log.error(px)
            return px
    if not len(px['x_avg']) == 6: 
        print(px)
        print(tabletitletext)
        return px
    table_titleless = remove_titles(table_poly[tablenum], cfgtable) # returns poly for specified tablenum but without 
    col_poly = column_poly(table_titleless, px, gcv_word, cfgtable)
    double_layered = identify_rows(col_poly, cfgtable['distance_th'], gcv_word, cfgtable)
    return layer_to_df(double_layered)

def parse_tables_img(image, gcv_word, pagenum = None, cfg=cfg):
    """
    takes an image and gcv_word and turns it a data frame. 
    arguments: 

    image: ndarray of the page of the image
    gcv_word: google cloud vision word level 
    pagenum: optional, will allow calculation of pos_id and adding of coordinates to the dataframe 
    cfg: defaults to config.yml. This function specifically does not use anything from it, 
        but it is passed into functions parse_table and modeled_layout, which both use it extensively. 
    """
    l = [] # sets list for later use, will become a list of dataframes 
    table_layout = modeled_layout(image, cfg=cfg, pagenum=pagenum) #uses function to get a modeled layout of the image using det2
    for i,x in enumerate(table_layout):
        df = parse_table(table_layout, gcv_word,i , cfgtable=cfg['Table']) # for each table in page, it passes it through parse_table to get dataframe
        df['x_1'] = x.coordinates[0] # gets coordinates of the table and adds them to dataframe 
        df['y_1'] = x.coordinates[1]
        df['x_2'] = x.coordinates[2]
        df['y_2'] = x.coordinates[3]
        if type(pagenum) == int: # If a pagenum was specified, it will calculate the position id for it and add it to df 
            log.info("pagenum {} was selected, calculating pos_id".format(pagenum))
            df['pagenum'] = pagenum
            df['pos_id'] = to_pos_id(
                y_1 = x.coordinates[1],
                y_2 = x.coordinates[3],
                pagenum = pagenum,
                docheight = image.shape[0] #calculates docheight using the height of the ndarray image 
            )
        if cfg['Table']['cont']['search']:
            df['cont'] = cont_or_not(x, gcv_word, cfgtable = cfg['Table'])
        l.append(df)
    df = pd.concat(l) # joins the data frames together into one, large data frame 
    df = df.reset_index() # rests the index so it is a normal index, leaves original index so you can tell the relative
#     position of each entry in reguards to its table
    if type(pagenum) == int: 
        df = df.sort_values(['pos_id', 'index']) # If pagenum specified, will sort the dataframes based on their relative pos_id in the larger df
    df = df.reset_index(drop=True)
    return df

def parse_page(pagenum, ocr_agent = None, cfg=cfg):
    """
    At the moment, is just a simple wrapper for parse_tables_img to allow you to easily specify each individual page and pdf from config. 
    Will likely expand later. 
    """
    dir = '{}/{}/Parsed_Tables/'.format(cfg['OUTPUT_DIRECTORY'], cfg['SOURCE_NAME'])
    csv = dir + '{}.csv'.format(pagenum)
   #dir = '{}/{}/Parsed_Tables/{}.csv'.format(cfg['OUTPUT_DIRECTORY'], cfg['SOURCE_NAME'], pagenum)
    if not os.path.exists(dir): 
            os.makedirs(dir)
            log.warning("Directories not created for this project, created them.")
    if(os.path.isfile(csv)): 
        log.info('File exists, returning from disk')
        return pd.read_csv(csv) 
    file = "{}/{}".format(cfg['INPUT_DIRECTORY'], cfg['SOURCE_PDF']) # Gets file position from inut directory and name set in config file 
    image = convert_PDF(file, pagenum)
    res, ocr_agent = gcv_response(image,pagenum, ocr_agent=ocr_agent, cfg=cfg) # Gets GCV stuff. 
    gcv_block, gcv_para, gcv_word, gcv_char = annotate_res(res, ocr_agent)
    df = parse_tables_img(image, gcv_word, pagenum, cfg=cfg) 
    df.to_csv(csv)
    log.info('Saved page {} to disk at {}'.format(pagenum, csv))
    return df 
   # return image, gcv_word
#def main():
    
#if __name__ == '__main__':
#   log.info('for debug only')
#   #main()
#   image = np.asarray(pdf2image.convert_from_path('/Users/liz/Documents/Projects/LayoutParser/test.pdf')[1])
#   table_layout = modeled_layout(image)
#   res, ocr_agent = gcv_response(image,1)
#   gcv_block, gcv_para, gcv_word, gcv_char = annotate_res(res, ocr_agent)
#   table_poly = to_polygons(table_layout)
#   table_txt = text_layout_from_selection(gcv_word, remove_titles(table_poly[0]))

#   #testing create polygons
#  # ll = create_bounding_polygons(remove_titles(table_poly[1]))
#  # hi = gcv_word.filter_by(ll[0], soft_margin = {"left":10, "right":10})
#  # lp.draw_box(image, ll, box_width=4).save("Tests/bruh3.png", "PNG")
#  # lp.draw_box(image, hi, box_width=4).save("Tests/bruh4.png", "PNG") 
#   table_title_1 = isolate_titles(table_poly[0])
#   tabletitletext = text_layout_from_selection(gcv_word, table_title_1)
#   
#   # %%
#   px = cols_px(tabletitletext)
#   l = remove_many_titles(table_poly)
#   a = column_poly(l[1], px, gcv_word)
#   lp.draw_box(image, a, box_width=4).save("Tests/21.png", "PNG")


# %%


# TODO
# fix ordering of title of bond
# find a way to associate rows with each other 
