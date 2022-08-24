import yaml
import subprocess
import sys

def open_y(): 
    conf=[]
    try :
        with open('base.yaml') as f:
            conf=yaml.load(f, Loader=yaml.SafeLoader)
    except Exception as e: 
        print('yaml 로딩 실패')
        print(e)
    return conf
    

def install_by_yaml(deps : list) :
    for d in deps :
        subprocess.check_call(['python', '-m', 'pip', 'install', d], shell=True)
        
install_by_yaml(open_y())




