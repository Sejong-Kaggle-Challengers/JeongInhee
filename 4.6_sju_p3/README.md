## 3월 30일 화요일 

### baseline : https://www.kaggle.com/craigmthomas/tps-mar-2021-stacked-starter

### 모델 : CNN(torch.hub.Vgg19_bn 모델 참고)
    class baseline_CNN(torch.nn.Module):
      def __init__(self):
      super(baseline_CNN,self).__init__() 

    self.layer1 = torch.nn.Sequential(
        torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
        torch.nn.BatchNorm2d(64),
        #torch.nn.GroupNorm(32,128),# group, channel
        torch.nn.ReLU(),
        torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
        torch.nn.BatchNorm2d(64),
        #torch.nn.GroupNorm(32,128),# group, channel
        torch.nn.ReLU(),
        #torch.nn.Dropout2d(0.3),
        torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0) #14*14
    )

   
    self.layer2 = torch.nn.Sequential(
        torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
        torch.nn.BatchNorm2d(128),
        #torch.nn.GroupNorm(32,128),# group, channel
        torch.nn.ReLU(),
        torch.nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
        torch.nn.BatchNorm2d(128),
        #torch.nn.GroupNorm(32,128),# group, channel
        torch.nn.ReLU(),
        #torch.nn.Dropout2d(0.3),
        torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0) #7*7
    )
    

 
    self.avgpool=torch.nn.AdaptiveAvgPool2d(output_size=(7,7))

    self.fc = torch.nn.Sequential(torch.nn.Linear(7*7*128,128,bias=True),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout2d(0.3),
                                  torch.nn.Linear(128,128,bias=True),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout2d(0.3),
                                  torch.nn.Linear(128,10,bias=True)
                                  )
  

    def forward(self,x):
      out = self.layer1(x)
      out = self.layer2(out)
      out=self.avgpool(out)
      out = out.view(out.size(0),-1)     
      out = self.fc(out)
      return out

#### 입력층과 출력층의 in_feautres와 out_features를 맞춰주는 것이 중요

### StratifiedKFold 

일반 KFold보다 고르게 분할함 ( from sklearn.model_selection import StratifiedKFold )
참고 : https://www.kaggle.com/mdmohibulhaquekhan/pytorch-densenet-stratified-kfold

## Group Normalization

batch크기에 크게 영향을 받는 batch normalization보다 강건함

https://blog.airlab.re.kr/2019/08/Group-Normalization

https://github.com/ppwwyyxx/GroupNorm-reproduce/blob/master/ImageNet-ResNet-PyTorch/resnet_gn.py

하지만 모든건 경험적이므로, 아닌 경우도 있음

[참고자료] https://nbviewer.jupyter.org/github/amaarora/amaarora.github.io/blob/master/nbs/Group%20Normalization.ipynb
