import window
import os
import cv2
####################### 여기까지가 원본 예제 import
import torch
import torchvision
from torchvision import models
import numpy as np

import time


ESC_KEY=27
FRAME_RATE = 5
SLEEP_TIME = 1/FRAME_RATE
# TODO 웹캠으로 바꾸기. 웹캠 달고 테스트 해보자!
camera = cv2.VideoCapture(0)
# 테스트용으로 꿀뷰를 사용해서 window_class 를 꿀뷰 클래스를 가져옴. 
capture = window.WindowCapture('HoneyviewClassX',None,FRAME_RATE) 

########################여기까지가 원본 변수들
device=torch.device('cuda:0')# 모델과 이미지를 cuda에 넘겨서 분석할거니까 변수 만들어두고

# CUDA 메모리가 넘칠때 쓰는 코드라고 했는데 좀 더 알아볼것
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 폴더 내에 모델을 둘거니까 주소 가져와서
HERE = os.path.dirname(os.path.abspath(__file__))
# 모델명 조인해서 model_path 완성
model_path=os.path.join(HERE,'best_model_3_134258.pth')

# 저장된 모델의 checkpoint(저장되던 당시의 모델의 weight 상태)를 저장해두고
checkpoint=torch.load(model_path)
# 모델의 형태는 기본 뼈대가 resnet34에 class(분류해야 할 카테고리)가 4개로 모델을 만들어와서
model = models.resnet34(num_classes=4)
# resnet34 모델과 형태가 좀 다르니까 strict=False로 설정해서 checkpoint에서의 가중치들을 씌워주고
model.load_state_dict(checkpoint, strict=False)
# 위에서 저장한 cuda로 모델을 넘겨서 gpu 연산이 되도록
model.to(device)

# Face detection XML load and trained model loading
emotion_classifier = model
EMOTIONS=['Embarassed', 'Pleasure','Rage','Sad']
model.to(device)


# Video capture using webcam

# 웹캠에서 전달되는 이미지를 계속 분석해야하니까 while문 안에서
while True:
    # 시간 췍
    start=time.time()
    # 꿀뷰 창에서 캡쳐한 이미지를 frame에 저장해서
    frame = capture.screenshot()
    # 이미지 분석을 위해 모델 input_shape에 맞춰 224*224 크기로 리스아지
    frame = cv2.resize(frame, (224, 224))
    # 좀 더 정확한 분석결과를 얻기 위해 normalize
    frame = frame.astype("float") / 255.0

    # cv에서 캡쳐한 이미지는 shape가 rgb(224(width),224(height),3)인데 모델이 받아들일 수 있는 이미지의 형태는 tensor형태(1(샘플 수), 3(채널 수), 224(width),224(height))
    # 따라서 모델에 넣기 전에 이미지를 변환해줘야됨. 변환해주기 위한 toTensor객체 만들어서(While문 밖으로 빼도 될것 같은데? TODO 정리하고 나서 빼보자)
    toTensor=torchvision.transforms.ToTensor()
    # 이미지 형태 변환
    frame=toTensor(frame)
    # 샘플 수 자리 반들어줘서 모양 확실히 맞춰주기!
    frame=frame.unsqueeze(0)
    # 변환했으면 cuda로 넘기자. 위에서 변환했던 float와 torch.float는 형태가 좀 다르니 변환도 같이
    frame=frame.to(device, dtype=torch.float)
    
    

    # 모델을 통해 예측한 값이 배열로 나오니까 
    preds=[]
    # tensorflow와 다르게 torch는 model이 train mode와 evaluation mode가 따로 있음. 그래서 예측하기 전에 evaluation mode로 변환해줘야됨
    model.eval()

    # no_grad를 안해주면 파이썬에서는 기본적으로 gradient(기울기)를 계산 해 주는데 evaluation mode에서는 기울기를 계산 할 필요가 없음. 기울기는 학습에서 쓰이는 개념
    # 그래서 eval()을 적용해 준 이후에 torch.no_grad()를 with statement로 감싸준다
    with torch.no_grad():
        # torch 모델의 prediction 방식. model(input)
        preds=model(frame)
        # prediction의 결과값 중 필요한 값들만 가져다 저장
        # TODO preds의 형태와 의미 알아볼 것
        _, predicted=torch.max(preds,1)
        # preds의 가장 큰 값의 인덱스를 가져와서 EMOTIONS 리스트에서 감정 불러오기
        predicted=EMOTIONS[preds[0].argmax(0)]
        
            # Assign labeling
    try:        
        # 불러온 감정을 frame에 띄워주기
        
        cv2.putText(frame, predicted, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 2)
        
        # TODO 비식별화 할 때 쓰려고 준비하던 코드. 감정분류 결과에 따라 같은 폴더 내의 PNG 이미지를 가져다가 bounding_box에 씌울 예정
        # img_path=f'{predicted}.png'
        # img_array=np.fromfile(img_path, np.uint8)
        # emo=cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # mask=emo.cv2.CV_8UC(np.uint8)

        # frame=cv2.copyTo(emo, mask, frame)
        
    except:
        pass


    
    cv2.putText(frame, predicted, (100,  100),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('Emotion Recognition', frame)
    
    delta= time.time()-start

    
    print('\r', 'preds=',preds[0], 'predicted=',predicted, 'Latency=',delta, end='')
   
    if delta <SLEEP_TIME:
        time.sleep(SLEEP_TIME-delta)

    # esc 누르면 끗
    key= cv2.waitKey(1) & 0xFF
    if key== ESC_KEY:
        break
    
    