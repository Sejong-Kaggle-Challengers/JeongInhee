## 11주차- 소설작가분류예측대회

### NLP 기본 개념

불용어 : 별 의미 없는 단어 (a, the 등)

표제어 : 기본 사전형 단어 , 단어의 기본형

형태소 : 의미를 가진 가장 작은 단위

한국어 전용 형태소 분석(token화) 라이브러리: Konlpy 

Padding : 모델에 입력하기 위해서는 일정한 길이를 가져야 하므로 진행

Embedding : 문장형태인 데이터를 vector화 하는 것과 같은 의미인 것 같음, 결론은 vectorizing=embedding

[embedding 관련 참고자료] : https://eda-ai-lab.tistory.com/428

### torchtext 

문장 데이터를 embedding 할 때, 유용하게 쓰이는 라이브러리

[documentation] : https://torchtext.readthedocs.io/en/latest/index.html

[참고자료]:https://simonjisu.github.io/nlp/2018/07/18/torchtext.html

[예제]:https://wikidocs.net/60314


