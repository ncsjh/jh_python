import window
import win32gui
import os
import cv2
####################### 여기까지가 원본 import
import torch
import torchvision
from torchvision import models
import dlib
import numpy as np

import time


ESC_KEY=27
FRAME_RATE = 5
SLEEP_TIME = 1/FRAME_RATE
# 테스트용으로 꿀뷰를 사용해서 window_class 를 꿀뷰 클래스를 가져옴. 
capture = window.WindowCapture('HoneyviewClassX',None,FRAME_RATE)

########################여기까지가 원본 변수들
device=torch.device('cuda:0')

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda')
HERE = os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(HERE,'best_model_3_134258.pth')


checkpoint=torch.load(model_path)
model = models.resnet34(num_classes=4)
model.load_state_dict(checkpoint, strict=False)
model.to(device)

# Face detection XML load and trained model loading
face_detection = dlib.get_frontal_face_detector()
emotion_classifier = model
EMOTIONS=['Embarassed', 'Pleasure','Rage','Sad']
model.to('cuda:0')
# Video capture using webcam
# camera = cv2.VideoCapture(0)

checkpoint=torch.load(model_path)
model = models.resnet34(num_classes=4)
model.load_state_dict(checkpoint, strict=False)
model.to(device)




while True:
    
    start=time.time()
    frame = capture.screenshot()
    
    # faces=face_detection(frame, 1)
    # print(faces)
    # face = sorted(faces, reverse=True, key=lambda x: (
    #         x[2] - x[0]) * (x[3] - x[1]))
    # (fX, fY, fW, fH) = face
    
    
    # For the largest image
    # try:
    #     emo_size=((fW), (fH))
    #     img_width, img_height=emo_size
    #     emo_pos=(fX+fW/2, fY+fH/2)
    #     emo_pos=((face.right()+face.left())/2, (face.top()+face.bottom())/2)
    #     (fX, fY, fW, fH) = face
    #     print(f'emosize={emo_size} , emo_pos={emo_pos}')
    # except Exception as e:
    #     print(e)
    roi = frame
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype("float") / 255.0

    tf_toTensor=torchvision.transforms.ToTensor()
    roi=tf_toTensor(roi)
    
    roi=roi.unsqueeze(0)
    roi=roi.to(device, dtype=torch.float)
    
    

    preds=[]
    model.eval()
    with torch.no_grad():
        preds=model(roi)
        _, predicted=torch.max(preds,1)
        predicted=EMOTIONS[preds[0].argmax(0)]
        
            # Assign labeling
    try:        
        cv2.putText(frame, predicted, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 2)
        # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
        img_path=f'{predicted}.png'
        img_array=np.fromfile(img_path, np.uint8)
        emo=cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # emo=cv2.resize(emo, (img_width, img_width))
        mask=emo.cv2.CV_8UC(np.uint8)

        # frame=cv2.seamlessClone(emo, frame, mask, emo_pos, cv2.NORMAL_CLONE)
        frame=cv2.copyTo(emo, mask, frame)
        
    except:
        pass


    
    cv2.putText(frame, predicted, (100,  100),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('Emotion Recognition', frame)
    
    delta= time.time()-start

    
    print('\r', 'preds=',preds[0], 'predicted=',predicted, 'Latency=',delta, end='')
   
    if delta <SLEEP_TIME:
        time.sleep(SLEEP_TIME-delta)
    key= cv2.waitKey(1) & 0xFF
    if key== ESC_KEY:
        break
    
    
    
    # 분류 및 할거 다 되는 코드 킵!
    # if len(faces) > 0:
    #     # For the largest image
    #     face = sorted(faces, reverse=True, key=lambda x: (
    #         x[2] - x[0]) * (x[3] - x[1]))[0]
    #     (fX, fY, fW, fH) = face
    #     # Resize the image to 48x48 for neural network

    #     roi = frame
    #     # roi = frame[fY:fY + fH, fX:fX + fW]
    #     roi = cv2.resize(roi, (224, 224))
    #     roi = roi.astype("float") / 255.0
        
    #     # roi = np.expand_dims(roi, axis=0)
    #     # roi=torch.utils.data.DataLoader(roi)
    #     # print("들어가기전:",type(roi[2][0][0]))
    #     # print("들어가기전 사이즈:",roi.shape)
        
    #         # Open two windows
    #     # Display image ("Emotion Recognition")
    #     # Display probabilities of emotion
    #     # cv2.imshow("Probabilities", canvas)
    #     # roi=np.expand_dims(roi, 1)
    #     # print("변환 중:",roi.shape)
    #     tf_toTensor=torchvision.transforms.ToTensor()
    #     roi=tf_toTensor(roi)
    #     # print("변환 중:",roi.size())
    #     # print(roi[0])
        
    #     roi=roi.unsqueeze(0)
    #     # print("변환 후:",type(roi))
    #     # print("변환 후:",roi.size())
    #     roi=roi.to(device, dtype=torch.float)
    #     # Emotion predict
        
        

    #     preds=[]
    #     model.eval()
    #     with torch.no_grad():
    #         preds=model(roi)
    #         _, predicted=torch.max(preds,1)
    #         predicted=EMOTIONS[preds[0].argmax(0)]
    #         # c=(predicted==EMOTIONS).squeeze()
    #         # print("predicted=",predicted)
            
    #             # Assign labeling
    #     cv2.putText(frame, predicted, (fX, fY - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 2)
    #     cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    #     # Label printing
    #     # for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
    #     #     text = "{}: {:.2f}%".format(emotion, prob * 100)
    #     #     w = int(prob[0] * 300)
    #     #     cv2.rectangle(canvas, (7, (i * 35) + 5),
    #     #                   (w, (i * 35) + 35), (0, 0, 255), -1)
    #     cv2.putText(frame, predicted, (100,  100),
    #     cv2.FONT_HERSHEY_SIMPLEX, 10, (100, 100, 100), 2)
    #     cv2.imshow('Emotion Recognition', frame)

        
    #     print('preds=',preds, 'predicted=',predicted, 'Latency=',delta)
    
################################ 원본 코드 킵
# while True:
#     start=time.time()
#     frame = capture.screenshot()
    
#     cv2.imshow("frame1",frame)
#     delta= time.time()-start
    
    
    
#     if delta <SLEEP_TIME:
#         time.sleep(SLEEP_TIME-delta)
#     key= cv2.waitKey(1) & 0xFF
#     if key== ESC_KEY:
#         break
