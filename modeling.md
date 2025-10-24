# modeling.ipynb 요약 및 이론 설명

목적  
- MNIST(손글씨 숫자, 28×28)를 대상으로 간단한 MLP(완전연결 신경망)으로 0~9 숫자 분류를 실습하고, 데이터 전처리 → 모델 정의 → 학습 → 평가까지 딥러닝 워크플로우를 이해하는 것이 목표.

파일 흐름(셀별 설명)
- 설명(마크다운)
  - 모델링의 직관: 회귀(Y = aX + b) → 복잡한 결정경계 학습(분류)
  - 활성화 함수(계단, 시그모이드, ReLU)와 다층 신경망(MLP) 소개
- 라이브러리 임포트
  - tensorflow, numpy, matplotlib 등
- 데이터 로드
  - (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
  - 결과: x_train shape (60000,28,28), y_train shape (60000,)
- 시각화/분포 확인
  - plt.imshow로 샘플 확인, np.unique로 라벨 분포 점검
- 정규화(스케일링)
  - x = x.astype('float32') / 255.0  → 픽셀값을 [0,1]로 변환 (학습 안정화)
- 모델 정의 (Keras Sequential MLP)
  - Flatten(input_shape=(28,28))  # 28×28 → 784
  - Dense(128, activation='relu')
  - Dense(64, activation='relu')
  - Dense(32, activation='relu')
  - Dense(10, activation='softmax')  # 클래스 확률 출력
- 컴파일
  - optimizer='adam'
  - loss='sparse_categorical_crossentropy'  # 정수 라벨 사용
  - metrics=['accuracy']
- 학습
  - model.fit(x_train, y_train, epochs=20, validation_data=(x_test,y_test))
  - history에 loss/accuracy 기록
- 학습 곡선 시각화
  - training/validation accuracy, loss 플롯으로 과적합·수렴 확인
- 예측 및 평가
  - pred = model.predict(x_test)  → 확률 벡터 반환
  - np.argmax(pred, axis=1)  → 예측 라벨
  - model.evaluate(x_test, y_test)  → [loss, accuracy]
  - sklearn.metrics.classification_report로 precision/recall/F1 확인

핵심 이론(간결한 수식과 설명)
- 완전연결층(Dense)
  - z^(l) = W^(l) a^(l-1) + b^(l)
  - a^(l) = φ(z^(l)) (φ: 활성화함수)
- ReLU
  - relu(z) = max(0, z)  → 비선형성 부여, 기울기 소실 문제 완화
- Softmax (출력 확률)
  - p_i = exp(z_i) / Σ_j exp(z_j)
- 교차엔트로피 손실(다중 클래스)
  - L = - log p_{y}  (sparse_categorical_crossentropy: y는 정수 라벨)
- 역전파와 최적화
  - 손실을 파라미터에 대해 미분(연쇄법칙) → 기울기 계산 → optimizer(예: Adam)로 가중치 갱신
- Adam 요약
  - 1차/2차 모멘트 추적으로 학습률을 파라미터별로 적응 조정 → 빠르고 안정적인 수렴 경향

실무 팁 및 주의사항
- MLP 한계: 이미지의 공간적 구조(인접 픽셀 관계)를 무시 → CNN이 일반적으로 더 적합
- 과적합 징후: train ↑, val ↓ 또는 val_loss 증가 → Dropout, L2, 데이터증강, EarlyStopping 권장
- 배치 크기: GPU 메모리와 성능에 맞춰 32/64 등 실험
- 학습률 스케줄러: ReduceLROnPlateau, StepDecay 등으로 일반화 개선
- 레이블 포맷: 정수 라벨이면 sparse_categorical_crossentropy, one-hot이면 categorical_crossentropy

간단한 개선/확장 코드 스니펫 (Keras)
- CNN으로 바꾸기 (채널 차원 추가 필요)
````markdown
// filepath: c:\Users\playdata2\Desktop\deep-learning\modeling_cnn_snippet.md

# CNN 예시 스니펫

x_train = x_train[..., None]  # (N,28,28,1)
x_test  = x_test[..., None]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

- 콜백 예시 (EarlyStopping + ModelCheckpoint)
````markdown
# 콜백 추가 예시
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ckpt = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

model.fit(..., callbacks=[es, ckpt])