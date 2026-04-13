from flask import Flask, request, jsonify
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import chromadb
from glob import glob
from tqdm import tqdm
import torch
import os

app = Flask(__name__)

# 추론에 CPU 사용
device = "cpu"

# ViT 모델 및 전처리기 로드
feature_extractor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
model.eval()  # 추론 모드 설정

# 영구 저장 ChromaDB 클라이언트 생성
client = chromadb.PersistentClient(path="../chroma_db")


@app.route("/build", methods=["POST"])
def build():
    """인덱스 초기 구축 (최초 1회 호출)"""

    # 이미 컬렉션이 존재하면 중복 생성 방지
    existing = [c.name for c in client.list_collections()]
    if "furniture" in existing:
        return jsonify({"message": "이미 인덱스가 존재합니다."})

    # hnsw:space = "cosine" 으로 설정해야 유사도가 0~1 사이로 정상 출력됨
    # 기본값(l2)은 유클리드 거리로 값이 수백~수천이 되어 유사도 계산 불가
    collection = client.create_collection(
        "furniture",
        metadata={"hnsw:space": "cosine"}
    )

    # 학습 이미지 전체 경로 수집
    img_list = sorted(glob("../img/train/*/*.jpg"))

    embeddings = []  # 이미지 벡터 목록
    metadatas = []   # 이미지 경로, 카테고리 등 부가 정보
    ids = []         # 각 항목의 고유 ID

    for i, img_path in enumerate(tqdm(img_list)):
        img = Image.open(img_path).convert("RGB")

        # os.path로 카테고리 추출 (Windows/Linux 경로 구분자 모두 대응)
        # 예) ../img/train/chair/00000004.jpg → "chair"
        cls = os.path.basename(os.path.dirname(img_path))

        # 이미지를 모델 입력 텐서로 변환
        img_tensor = feature_extractor(images=img, return_tensors="pt").to(device)

        # 추론 시 그래디언트 계산 비활성화 (메모리/속도 최적화)
        with torch.no_grad():
            outputs = model(**img_tensor)

        # pooler_output: 이미지 전체를 대표하는 벡터 [1, 384] → [384]
        embedding = outputs.pooler_output.detach().cpu().numpy().squeeze().tolist()
        embeddings.append(embedding)

        # URI를 웹에서 접근 가능한 경로로 정규화
        # ../img/train/chair/00000004.jpg → img/train/chair/00000004.jpg
        uri = img_path.replace("\\", "/")   # Windows 역슬래시를 슬래시로 변환
        if uri.startswith("../"):
            uri = uri[3:]                   # 앞의 "../" 제거

        metadatas.append({"uri": uri, "name": cls})
        ids.append(str(i))

    # 수집한 모든 데이터를 DB에 저장
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
    return jsonify({"message": f"완료: {len(img_list)}개"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
