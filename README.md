Soma 7th backend 과제
=====================

train_rf.py : 학습
  (rf는 과제 제출 중간에 시도했던 방법인 random forest의 약자, 실제 과제와는 무관)

server.py : FLASK 기반 기초적인 REST 서버



mecab을 이용해 상품 제목을 형태소 분석하고, 이를 CountVectorizer로 센 뒤,
Tk-idf 방법을 이용해 빈도수 분석을 정교하게 해서 SVM에 넣었습니다.

이미지는 처리하지 않았습니다.

