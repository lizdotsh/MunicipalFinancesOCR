import numpy as np 
import pdf2image
import pandas as pd
import os
import sys
import io
import logging
import re
import layoutparser as lp
from PIL import Image
from pyparsing import col
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Processing")
import yaml
with open('config.yml') as f:
    try:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print(dict)
    except yaml.YAMLError as e:
        print(e
import ocr as o
import det2 as d2
import processing as p

# This file finds the titles in the OCR'd PDF and returns a list of the titles and PosID's

def extract_title(table_layout, gcv_block):
    """
    Need to combine in cases where gcv splits it across two things. 
    """
    df = table_layout.to_dataframe()
    df_titles = df[df['text'].str.contains('History :') == True]
    kjk = df_titles["text"].str.split("History :", expand=True)
    return df_titles


# TODO
# Need to combine in cases where gcv splits it across two things.
# Search whole thing looking for that 
# fix Pos ID function to work better