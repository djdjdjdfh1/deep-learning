# model8.ipynb — 초심자용 설명서

간단 요약
- 이 노트북은 ResNet(Residual Network) 개념을 손수 구현하고, 간단한 학습/평가 루프까지 보여줍니다.
- 목적: 스킵 커넥션(Residual Block)이 무엇인지, 왜 쓰는지, 그리고 간단한 ResNet 구조를 직접 만들어 보는 것.

목차
1. 핵심 개념
2. 주요 코드 설명
3. 학습 및 평가 흐름
4. 실행 방법 (Windows)
5. 자주 발생하는 문제와 디버깅 팁
6. 다음 학습 단계

1. 핵심 개념 (초심자용)
- Convolution (Conv2d): 이미지에서 특징을 추출하는 필터입니다.
- BatchNorm: 각 배치의 분포를 정규화해 학습을 안정화합니다.
- ReLU: 비선형 활성화 함수로 음수를 0으로 만듭니다.
- Skip connection (스킵 커넥션): 블록의 입력을 출력에 더해주는 연결. 깊은 네트워크에서 기울기 소실을 줄이고 학습을 쉽게 합니다.
  - 수식으로 보면 H(x) 대신 F(x)+x를 학습하도록 바꾸는 것과 같습니다.
  - 입력/출력 채널이나 해상도가 다르면 1x1 conv(프로젝션)를 사용해 차원을 맞춥니다.

2. 주요 코드 설명
- BasicBlock 클래스:
  - 내부에 conv -> batchnorm -> ReLU -> conv -> batchnorm 구조가 있고,
  - 입력 x를 downsample(1x1 conv)로 차원을 맞춘 뒤 연산 결과에 더합니다.
  - 마지막에 ReLU를 다시 적용해 블록 출력을 얻습니다.
- ResNet 클래스:
  - 여러 BasicBlock을 쌓고 평균 풀링(AveragePool)으로 크기를 줄인 뒤(예시) 완전연결층(FC)으로 분류하려는 구조입니다.
  - 현재 예제는 작은 입력(32x32)을 가정합니다.

3. 학습 및 평가 흐름
- 데이터: torchvision.datasets.ImageFolder를 사용해 `./data/train`, `./data/test`에서 불러옵니다.
- transform: Resize(32,32) → ToTensor → Normalize
- DataLoader: batch_size=5 (예시)
- Optimizer: Adam(lr=1e-3), Loss: CrossEntropyLoss
- 학습: epoch 루프에서 forward, loss.backward(), optim.step() 수행
- 평가: model.eval() 및 with torch.no_grad()로 loss와 accuracy 계산

4. 실행 방법 (Windows)
- 필요한 패키지 설치:
  pip install torch torchvision tqdm jupyterlab
- Jupyter로 열기:
  cd c:\Users\playdata2\Desktop\deep-learning
  jupyter notebook model8.ipynb
- 또는 VS Code에서 노트북 파일 열기

5. 자주 발생하는 문제 & 팁
- shape mismatch: 스킵 연결에서 입력과 출력의 채널/크기가 다르면 더할 수 없습니다. 이 노트북에서는 downsample(1x1 conv)로 맞춤.
- batch_size가 너무 작으면 BatchNorm 동작이 불안정할 수 있음.
- 모델 저장/로드: torch.save(model.state_dict(), 'resnet.pth') 후 load_state_dict 시 map_location 사용 (CPU/GPU 차이).
- 학습이 불안정하면 learning rate를 줄이거나 배치 사이즈 늘리기.

6. 다음 학습 단계 (권장)
- torchvision.models.resnet의 사전학습된 모델(resnet18/resnet50) 사용해보기.
- BasicBlock을 쌓아 깊이가 다른 ResNet(예: ResNet18) 구성해보기.
- 데이터 증강(augmentation) 적용해보기.
- 실험: Skip connection이 없는 모델과 비교해 학습 곡선(accuracy, loss) 관찰하기.

간단한 확인 코드(노트북에서 실행해볼 것)
```python
import torch
sample = torch.randn(1,3,32,32)
from model8 import ResNet  # 노트북 코드가 모듈화된 경우
model = ResNet(2)
out = model(sample)
print(out.shape)