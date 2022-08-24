import os
import shutil
import numpy as np
from glob import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

HERE=os.path.dirname(os.path.abspath(__file__))

def extCls(dir):
    cls=os.listdir(HERE)
    cls.remove('resize_nas.py')
    
    print(cls)
    return cls 

def resize(cls : list) :
    try:
        for cl in cls:
            path=os.path.join(HERE,cl)
            print(f'경로 : {path}')
            print(f'{cl} 폴더 내 파일 불러오기')
            li=os.listdir(path)
            print(f'{cl} 폴더 내 파일 리사이즈 시작')
            w=len(li)
            c=0
            for file in li:
                try:
                    c+=1
                    print('\r', int(c/w*100),'%', '  ', c,'/',w, file, end='')
                    dest_dir=os.path.join(HERE,'resize_600',cl)
                    dest_file=os.path.join(HERE,'resize_600',cl, file)
                    
                    f=os.path.join(HERE,cl, file)
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)
                        print(f'{dest_dir} 폴더가 생성됐습니다')
                    img=Image.open(f)
                    img=img.resize((600,600))
                    img.save(dest_file)
                except OSError as e:
                    print(file, ' 파일에서 문제가 발생했습니다.')
                    print(e)
    except OSError as e:
        print('경로가 잘못되었습니다')
        print(e)
            
            
resize(extCls(HERE))
