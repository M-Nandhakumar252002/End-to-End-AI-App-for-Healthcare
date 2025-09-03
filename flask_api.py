from flask import Flask, request, jsonify
from utils import load_model, read_and_preprocess, make_gradcam_heatmap, overlay_heatmap_on_image
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
model = load_model("model/model.keras")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    arr, raw = read_and_preprocess(file.read())
    prob = float(model.predict(arr)[0][0])
    pred_label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"

    heatmap = make_gradcam_heatmap(arr, model)
    overlay = overlay_heatmap_on_image(heatmap, raw)

    # Convert overlay to base64
    pil_img = Image.fromarray(overlay)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64_overlay = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({"prediction": pred_label, "probability": prob, "gradcam": b64_overlay})

if __name__ == "__main__":
    app.run(debug=True)
