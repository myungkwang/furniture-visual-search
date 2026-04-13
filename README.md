# 🪑 Furniture Visual Search

사진을 찍으면 가장 유사한 가구를 찾아주는 이미지 유사도 검색 웹 애플리케이션입니다.  
**MSA(Microservice Architecture)** 구조로 설계되어 각 기능이 독립적인 서비스로 분리되어 있습니다.

---

## 📱 데모

| 이미지 업로드 
| 검색 결과 |
|:---:
|:---:
|
| (스크린샷) 
| (스크린샷) 
|

---

## 🏗️ 아키텍처

모바일/브라우저
↓
Gateway (5000) ← 요청 수신 및 라우팅
↓ ↓
Embedding (5001) Search (5002) ← 이미지 벡터화 / 유사도 검색

Index (5003) ← DB 인덱스 구축 (최초 1회)
↓
ChromaDB ← 벡터 영구 저장


---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| 백엔드 | Python, Flask |
| AI 모델 | Facebook DINO ViT-S/16 (HuggingFace) |
| 벡터 DB | ChromaDB (Cosine Similarity) |
| 프론트엔드 | HTML, CSS, Jinja2 |
| 아키텍처 | MSA (Microservice Architecture) |

---

## 📁 프로젝트 구조

furniture-visual-search/
├── gateway/ # 5000 - API 게이트웨이 + 프론트엔드
│ ├── app.py
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ └── static/uploads/
├── embedding_service/ # 5001 - 이미지 벡터화
│ └── app.py
├── search_service/ # 5002 - ChromaDB 유사도 검색
│ └── app.py
├── index_service/ # 5003 - 인덱스 구축
│ └── app.py
├── chroma_db/ # 벡터 DB 저장소
└── img/train/ # 학습 이미지
├── chair/
├── sofa/
└── ...


---

## ⚙️ 설치 및 실행

1. 패키지 설치
```bash
pip install flask flask-cors transformers torch Pillow chromadb requests tqdm

2. 인덱스 구축 (최초 1회)
# 터미널 1
cd index_service && python app.py

# 터미널 2 - 빌드 요청
python -c "import requests; print(requests.post('http://localhost:5003/build').json())"

3. 서비스 실행
cd embedding_service && python app.py  # 터미널 3
cd search_service    && python app.py  # 터미널 4
cd gateway           && python app.py  # 터미널 5

4. 접속
PC: http://localhost:5000
모바일 (같은 와이파이): http://[서버IP]:5000

🔍 동작 방식
1. 사용자가 모바일로 가구 사진 촬영 또는 갤러리에서 선택
2. Gateway가 이미지를 수신하여 Embedding Service로 전달
3. Embedding Service가 DINO ViT 모델로 이미지를 384차원  벡터로 변환
4. Search Service가 ChromaDB에서 Cosine 유사도로 가장 가까운 이미지 3개 검색
5. 유사도 순서대로 결과 화면에 표시
📌 참고사항
GPU 없이 CPU만으로 동작합니다
인덱스 구축 시 이미지 수에 따라 시간이 소요됩니다
모바일 접속은 PC와 동일한 와이파이 환경에서만 가능합니다