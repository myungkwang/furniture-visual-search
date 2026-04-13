from flask import Flask, request, jsonify
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import torch

app = Flask(__name__)

device = "cpu"

# 모델 호출
feature_extractor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
model.eval()

# 이미지가 들어오면 저장된 이미지를 불러와서 임베딩
@app.route("/embed", methods=["POST"])
def embed():
    file = request.files.get("image")
    img = Image.open(file).convert("RGB")

    img_tensor = feature_extractor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**img_tensor)

    embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze().tolist()
    return jsonify({"embedding": embedding})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
