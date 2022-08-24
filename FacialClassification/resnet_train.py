
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil


device = torch.device("cuda:0") 

# 모델 학습 및 검증에 적합한 형태로 resize, normalize

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



# TODO 주소 입력하는 부분. 수정해야됨
HERE=os.path.dirname(os.path.abspath(__file__))
data_dir = r'C:\Users\user\python\FacialClassification'
train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train','resize_600'), transforms_train)
test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms_test)
# 여기까지! 수정!!


# TODO 배치사이즈는 이전 노트북 기준으로 8 잡았던거니까 좀 더 키워봐도 될듯!
# 데이터를 폴더채로 불러오고 폴더명으로 class_name까지 해주는 방법.
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=8, shuffle=True, num_workers=1)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=8, shuffle=True, num_workers=1)


print('학습 데이터셋 크기:', len(train_datasets))
print('테스트 데이터셋 크기:', len(test_datasets))

class_names = train_datasets.classes
print('클래스:', class_names)

model = models.resnet34(pretrained=True)
# TODO nn.Linear, model.fc 좀 더 알아보기
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# 쌩 처음부터 학습시킬땐 optimizer를 Adam 쓰는게 더 좋은데
# Transfer Learning에는 SGD(경사하강법)가 좋음
# TODO 추후 python resnet_train.py --epochs=10? 이런식으로 설정할 수 있도록 수정해보고
# Default는 이정도로
# learning rate, momentum, epochs 수정할 수 있게 만들고
num_epochs = 12
lr=0.001
momentum=0.9

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)




model.train()
start_time = time.time()

# 정확도가 제일 잘 나오는 모델 나오면 저장하기 위한 초기화
best_model = model
best_accuray = 0.0

# 입력한 에포크만큼 학습시키는 부분

# 학습 진행률 안보이면 답답하니까 전체 길이 저장해뒀다가
full_len=len(train_dataloader)

for epoch in range(num_epochs):
    # 이전 에포크 학습시키고 남아있는 running_loss, corrects 초기화 시켜주고
    running_loss = 0.
    running_corrects = 0
    epoch_start_time=time.time()

    # 전체 데이터 중에 학습에 이용된 데이터 숫자 초기화
    trained=0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 학습된 데이터 숫자 카운트 
        trained+=1
        # 학습된 비율을 구하고
        p_rate=round(trained/full_len,4)

        # 루프가 돌고 나면 optimizer에 grad값이 남아있는데 이걸 이대로 역전파 시키면
        # 학습이 제대로 안돼서 gradient값을 0으로 만들어주는 .zero_grad() 사용
        optimizer.zero_grad()

        # 모델에 이미지를 넣어서 결과를 예측하고
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # 그 값이랑 라벨링 된 값이랑 비교해서 loss값을 획득
        loss = criterion(outputs, labels)

        # TODO 딥러닝 모델 학습 과정에서 loss.backward()를 호출하면 각 파라미터들의 gradient값의 변화도가 저장된다고 하는데
        # 좀 말이 좀 애매한것 같으니까 이것도 알아보기
        loss.backward()
        # loss값을 통해 설정된 학습률 만큼 weights를 수정. 
        optimizer.step()

        # 시간이 얼마나 걸렸는지, 예상시간은 얼마나 되는지 보고싶어서 넣은 코드
        time_lapse=round(time.time()-epoch_start_time, 1)
        estimated_remain_time=round(time_lapse/p_rate-time_lapse,1)

        # 자꾸 콘솔창에 지저분하게 남아서 없애려고 공백으로 새로고침 한번 해줌
        print('\r', '                                                                                                             ', end='')
        
        print('\r', f'epoch : {epoch} / {num_epochs}, {round(p_rate*100, 2)}% , time_lapse : {time_lapse}, estimated_remain_time : {estimated_remain_time}', end='')

        # loss값과 corrects값을 계산
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # 한 에포크가 다 돌고 나서 이전보다 정확도가 좋아지면 그 모델 저장. 
        # 중간에 학습을 멈추는 경우가 생길 수 있으니까 그때그때 저장 해놓고
        # 그 모델의 state를
        # .model 파일이랑 .pth 두가지로 저장해서 필요할때 써먹자

    if best_accuray < running_corrects :
        best_model=model
        torch.save(best_model.state_dict(), f'best_{epoch}_{running_corrects}.model')
        torch.save(best_model.state_dict(), f'best_model_{epoch}_{running_corrects}.pth')

    # 에포크 진행중의 학습률이랑 loss, accuracy, 진행시간 표시
    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.
    print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
    
# 모든 에포크가 끝나면 
print ("The best model has an accuracy of " + str(best_accuray))
torch.save(best_model.state_dict(), 'best_1.model')
torch.save(best_model.state_dict(), 'best_model_1.pth')