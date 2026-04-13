from flask import Flask, request, jsonify
import chromadb

app = Flask(__name__)

# 공유 DB 폴더 참조 (index_service가 만든 DB를 그대로 읽음)
client = chromadb.PersistentClient(path="../chroma_db")

# 앱 시작 시점에 컬렉션을 가져오지 않음
# → 요청이 들어올 때마다 가져와서 index_service 실행 순서에 무관하게 동작
collection = None


@app.route("/search", methods=["POST"])
def search():
    """쿼리 임베딩을 받아 ChromaDB에서 유사한 이미지 검색"""

    global collection

    # 컬렉션이 아직 로드되지 않은 경우 이 시점에 가져오기
    # index_service에서 build가 완료된 후에만 정상 동작함
    if collection is None:
        try:
            collection = client.get_collection("furniture")
        except Exception:
            # 아직 인덱스가 빌드되지 않은 경우 안내 메시지 반환
            return {"error": "인덱스가 아직 준비되지 않았습니다. index_service에서 /build를 먼저 실행해주세요."}, 503

    data = request.json
    embedding = data["embedding"]

    # ChromaDB에서 가장 유사한 이미지 3개 검색
    # cosine 거리: 0(동일) ~ 2(정반대), 값이 작을수록 유사
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=["metadatas", "distances"]
    )

    items = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        # cosine 거리를 유사도 점수로 변환
        # dist = 0 → score = 1.0 (완전 동일)
        # dist = 1 → score = 0.0
        # dist = 2 → score = -1.0 (정반대, 실제로는 거의 발생 안 함)
        # 0~100% 백분율로 표시하기 위해 100을 곱함
        score = round((1 - dist) * 100, 1)

        items.append({
            "uri": meta["uri"],      # 이미지 파일 경로
            "name": meta["name"],    # 카테고리 이름
            "score": score           # 유사도 (0~100%)
        })

    return jsonify({"results": items})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
