from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # 모바일 크로스도메인 접속 허용

# Embedding, Search 서비스 주소
EMBEDDING_URL = "http://localhost:5001/embed"
SEARCH_URL    = "http://localhost:5002/search"

# 업로드 이미지 저장 경로 (static 하위여야 HTML에서 접근 가능)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 학습 이미지 루트 경로 (gateway 기준 상위 폴더의 img/)
TRAIN_IMG_ROOT = os.path.join(os.path.dirname(__file__), "..", "img")


@app.route("/")
def index():
    """메인 페이지: 카메라/갤러리 업로드 폼"""
    return render_template("index.html")


@app.route("/train-img/<path:img_path>")
def serve_train_img(img_path):
    """
    학습 이미지를 브라우저에 제공하는 라우트
    - ChromaDB에 저장된 URI는 'img/train/chair/00000004.jpg' 형식
    - /train-img/img/train/chair/00000004.jpg 로 요청이 들어오면
      실제 파일 경로로 변환하여 반환
    """
    # img_path 예) "img/train/chair/00000004.jpg"
    # TRAIN_IMG_ROOT 는 "../img" 이므로 "img/" 앞부분을 제거하고 연결
    relative = img_path.replace("img/", "", 1)          # "train/chair/00000004.jpg"
    full_path = os.path.abspath(os.path.join(TRAIN_IMG_ROOT, relative))
    return send_file(full_path)                          # 파일을 HTTP 응답으로 전송


@app.route("/search", methods=["POST"])
def search():
    """업로드된 이미지를 받아 유사 이미지 검색 후 결과 페이지 반환"""

    # 폼에서 전송된 이미지 파일 가져오기
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "이미지 없음"}), 400

    # 업로드 이미지를 서버에 저장
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Embedding Service에 이미지 전송 → 벡터 추출 요청
    with open(save_path, "rb") as f:
        emb_response = requests.post(EMBEDDING_URL, files={"image": f})
    embedding = emb_response.json()["embedding"]

    # Search Service에 벡터 전송 → 유사 이미지 검색 요청
    search_response = requests.post(SEARCH_URL, json={"embedding": embedding})
    results = search_response.json()["results"]

    # 업로드 이미지의 웹 접근 경로 (static/ 하위이므로 바로 사용 가능)
    query_img_url = "/" + save_path.replace("\\", "/").replace(
        os.path.dirname(__file__).replace("\\", "/") + "/", ""
    )

    return render_template("result.html",
                           query_img=query_img_url,
                           items=results)


if __name__ == "__main__":
    # host="0.0.0.0": 같은 네트워크의 모바일 기기에서 접속 가능
    app.run(host="0.0.0.0", port=5000, debug=True)
