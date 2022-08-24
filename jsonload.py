import numpy as np
import matplotlib.pyplot as plt
import shutil, json
import os.path as osp
import os
from collections import Counter

HERE = osp.dirname(osp.abspath(__file__))
RAWDIR = osp.join(HERE, 'raw')
LABELDIR = osp.join(HERE, 'label')
FILENAME= 'img_emotion_sample_data'
SYNTAX = '.json'

ANNOTATIONDICT = {'annot_A', 'annot_B', 'annot_C'}

def extractCategories(directory) -> list[str]:
    categories_list = [i for i in os.listdir(directory) if os.path.isdir(osp.join(directory, i))]
    return categories_list

def sortByCategory(categories_list: list):
    BINDIR = osp.join(HERE, 'bin')
    TESTEDDIR = osp.join(HERE, 'tested')

    if not osp.exists(BINDIR):
        os.mkdir(BINDIR)
    if not osp.exists(TESTEDDIR):
        os.mkdir(TESTEDDIR)
    
    for i in categories_list:
        if not osp.exists(osp.join(TESTEDDIR, i)):
            os.mkdir(osp.join(TESTEDDIR, i))

    for i in categories_list:
        base_dir = FILENAME + '(' + i + ')' + SYNTAX
        json_file_dir = osp.join(LABELDIR, base_dir)
        with open(json_file_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for eachfile in data:
                RES = [eachfile[j]['faceExp'] for j in [i for i in eachfile.keys() if i in ANNOTATIONDICT]]
                filedir = osp.join(RAWDIR + '/' + i, eachfile['filename'])

                if len(set(RES)) > 1:
                    destination_dir = BINDIR
                    RES_dict = {}
                    for k in RES:
                        if k =='중립':
                            continue
                        if k in RES_dict:
                            RES_dict[k] += 1
                        else:
                            RES_dict[k] = 1
                    if max(RES_dict.values()) > len(RES_dict)//2:
                        destination_dir = osp.join(TESTEDDIR, max(RES_dict, key=RES_dict.get))
                else:
                    destination_dir = osp.join(TESTEDDIR, RES[0])
                shutil.move(filedir, destination_dir)

sortByCategory(extractCategories(RAWDIR))