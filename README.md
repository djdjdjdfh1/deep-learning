[ 수업 준비 체크리스트 ]

설정 / 앱 에서 Microsoft Visual Studio Code, Miniconda3를 제거합니다.

사용자계정 폴더의 .conda, .ipython, .vscode, miniconda3 폴더와 .condarc 파일을 제거합니다. 아울러 C:\Users\사용자계정\AppData\Roaming\Code 폴더도 제거합니다.



윈도우 검색창에서 anaconda prompt를 실행한수 다음과정을 순차적으로 실행

```
# 현재 가상환경 비활성화(만약 deep이 활성화 되어 있다면)
conda deactivate

# 가상환경 확인하기
conda env list

# 설치된 가상환경이 만약 deep 가 있다면 삭제 - 없으면 안해도 됨
conda env remove -n deep

# python 3.11 버전으로 가상환경 설치하면서 필요한 패키지 동시 설치
conda create -n deep python=3.11 pytorch torchvision -c pytorch

# deep 가상환경 활성화
conda activate deep

# 설치확인
conda list   --> 모든라이브러리 확인(찾기 힘들) 비추

# 라이브러리명으로 설치리스트에 찾는 명령어
conda list | findstr "torch"   -> 출력이 나오는지 확인

conda list | findstr "torchvison"

# VScode 설치후 -> 노트북에서 커널선택 하면 ... 우리가만든 deep 이름의 가상환경이 존재
해당 가상환경 선택해서 실행



```








파이썬 머신러닝 판다스 데이터분석 교재 소스코드

https://github.com/tsdata/pandas-data-analysis

---


시각화 


https://matplotlib.org/cheatsheets/
