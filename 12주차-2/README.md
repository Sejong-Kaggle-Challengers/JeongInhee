## 12주차 금요일

### baseline : https://www.kaggle.com/craigmthomas/tps-mar-2021-stacked-starter

### target encoder

[여러가지 encoder] https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark

[Leaveoneout encoder] https://brendanhasz.github.io/2019/03/04/target-encoding#target-encoding

### StratifiedKFold 

일반 KFold보다 고르게 분할함 ( from sklearn.model_selection import StratifiedKFold )

### Stacking 

kaggle Stacking example - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python 

기본적인 Stacking - https://lsjsj92.tistory.com/558

CV 기반 이용한 Stacking - https://lsjsj92.tistory.com/559?category=853217


#### Stacking Level 1 model:

xgboost, lightGBM, catboost, RidgeClassifier, SGDClassifier, HistGradintBoostingClassifier

#### Stacking Level 2 model :

RidgeClassifier
