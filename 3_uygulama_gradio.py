import json
import time
from pathlib import Path
import cv2
from PIL import Image, ImageDraw
import gradio as gr
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_yolov8n.pt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

model = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt")

TR_NAMES = {
    "person": "person (İnsan)", "car": "car (Araba)", "cat": "cat (Kedi)",
    "dog": "dog (Köpek)", "bottle": "bottle (Şişe)", "cup": "cup (Bardak/Kupa)",
    "chair": "chair (Sandalye)", "couch": "couch (Koltuk)", "laptop": "laptop (Dizüstü Bilgisayar)",
    "mouse": "mouse (Fare)", "keyboard": "keyboard (Klavye)", "cell phone": "cell phone (Cep Telefonu)",
    "book": "book (Kitap)", "clock": "clock (Saat)", "backpack": "backpack (Sırt Çantası)",
    "handbag": "handbag (El Çantası)", "tv": "tv (Televizyon/Monitör)", "bowl": "bowl (Kase)",
    "banana": "banana (Muz)", "apple": "apple (Elma)", "pizza": "pizza (Pizza)",
    "fork": "fork (Çatal)", "knife": "knife (Bıçak)", "spoon": "spoon (Kaşık)",
    "potted plant": "potted plant (Saksı Bitkisi)", "bed": "bed (Yatak)", "dining table": "dining table (Masa)"
}

def tekli_tahmin(image: Image.Image, conf_thresh: float, iou_thresh: float):
    if image is None:
        return None, "⚠️ Lütfen bir fotoğraf yükleyin.", []

    t0 = time.perf_counter()
    results = model.predict(source=image, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    res = results[0]
    plotted_bgr = res.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    rows = []
    if res.boxes is not None and len(res.boxes) > 0:
        for i, box in enumerate(res.boxes, 1):
            cls_id = int(box.cls[0].item())
            raw_name = res.names[cls_id]
            tr_name = TR_NAMES.get(raw_name, raw_name)
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            rows.append([i, tr_name, f"%{conf*100:.2f}", f"[{x1}, {y1}, {x2}, {y2}]", f"{infer_ms:.1f} ms"])

        ozet = (
            f"✅ **{len(res.boxes)} adet nesne tespit edildi!** | "
            f"Tahmin Süresi (Inference Time): **{infer_ms:.2f} ms**"
        )
    else:
        ozet = f"⚠️ Nesne tespit edilemedi (Güven eşiğini düşürmeyi deneyin). | Süre: **{infer_ms:.2f} ms**"

    return Image.fromarray(plotted_rgb), ozet, rows

def bes_fotograf_degerlendir(img1, gt1, img2, gt2, img3, gt3, img4, gt4, img5, gt5, conf_thresh):
    items = [
        (img1, gt1, "Fotoğraf 1"),
        (img2, gt2, "Fotoğraf 2"),
        (img3, gt3, "Fotoğraf 3"),
        (img4, gt4, "Fotoğraf 4"),
        (img5, gt5, "Fotoğraf 5"),
    ]
    gallery_list = []
    saved_pil_list = []
    md_lines = [
        "### 📊 Çoklu Görüntü Karşılaştırma Tablosu",
        "| Fotoğraf | Gerçek Nesne (Ground Truth) | Modelin Tahmin Ettiği Nesneler | En Yüksek Güven Skoru | Tahmin Süresi | Başarı Durumu |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for idx, (img, gt_text, label) in enumerate(items, 1):
        if img is None:
            continue
        t0 = time.perf_counter()
        results = model.predict(source=img, conf=conf_thresh, imgsz=640, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        res = results[0]
        plotted_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
        out_pil = Image.fromarray(plotted_rgb)
        out_pil.save(RESULTS_DIR / f"telefon_foto_{idx}_sonuc.jpg", quality=92)

        detected_names = []
        top_conf = 0.0
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                c_id = int(b.cls[0].item())
                c_name = res.names[c_id]
                detected_names.append(TR_NAMES.get(c_name, c_name))
            top_conf = float(res.boxes[0].conf[0].item())
            pred_str = ", ".join(dict.fromkeys(detected_names))
        else:
            pred_str = "Tespit Edilemedi"

        gt_clean = (gt_text or "").strip().lower()
        is_ok = any(gt_clean in d.lower() for d in detected_names) if gt_clean else (len(detected_names) > 0)
        durum = "✅ Başarılı" if is_ok else "❌ Başarısız / Kısmi"

        caption = f"{label}: Tahmin={pred_str} (%{top_conf*100:.1f})"
        gallery_list.append((out_pil, caption))
        saved_pil_list.append((out_pil, label, gt_text, pred_str, top_conf))

        md_lines.append(
            f"| **{label}** | {gt_text or '-'} | {pred_str} | %{top_conf*100:.2f} | {infer_ms:.1f} ms | {durum} |"
        )

    if saved_pil_list:
        cw, ch = 280, 300
        canvas = Image.new("RGB", (cw * len(saved_pil_list), ch), (245, 245, 245))
        for i, (im, lbl, gt_t, pr_t, cf) in enumerate(saved_pil_list):
            im_r = im.resize((cw - 16, ch - 50))
            canvas.paste(im_r, (i * cw + 8, 8))
            draw = ImageDraw.Draw(canvas)
            draw.text((i * cw + 12, ch - 36), f"{lbl} | Gercek: {gt_t}", fill=(20, 20, 20))
            draw.text((i * cw + 12, ch - 18), f"Tahmin: {pr_t[:22]} (%{cf*100:.0f})", fill=(0, 100, 40))
        canvas.save(RESULTS_DIR / "telefon_5_fotograf_sonuc.png")

    return gallery_list, "\n".join(md_lines)

def metrikleri_oku():
    m_path = RESULTS_DIR / "metrikler.json"
    if not m_path.exists():
        return "Metrik dosyası bulunamadı."
    m = json.loads(m_path.read_text(encoding="utf-8"))
    return f"""
### 🏆 Model Performans Metrikleri (Test Kümesi)
| Metrik / Parametre | Değer |
| :--- | :--- |
| **Model Mimarisi** | `{m.get('Model')}` |
| **Giriş Boyutu (Input Size)** | `{m.get('Input_Size')}` |
| **Optimizer & Loss** | `{m.get('Optimizer')}` / `{m.get('Loss_Function')}` |
| **mAP@50** | **%{m.get('mAP50', 0)*100:.2f}** (`{m.get('mAP50')}`) |
| **mAP@50-95** | **%{m.get('mAP50_95', 0)*100:.2f}** (`{m.get('mAP50_95')}`) |
| **Ortalama IoU (Mean IoU)** | **%{m.get('Mean_IoU', 0)*100:.2f}** (`{m.get('Mean_IoU')}`) |
| **Precision (Kesinlik)** | **%{m.get('Precision', 0)*100:.2f}** (`{m.get('Precision')}`) |
| **Recall (Duyarlılık)** | **%{m.get('Recall', 0)*100:.2f}** (`{m.get('Recall')}`) |
| **F1-Score** | **%{m.get('F1_Score', 0)*100:.2f}** (`{m.get('F1_Score')}`) |
| **Eğitim Süresi** | `{m.get('Training_Time_Sec')} saniye` |
| **Ortalama Tahmin Süresi** | `{m.get('Inference_Time_Ms')} ms / görüntü` |
"""

with gr.Blocks(title="YOLOv8 Nesne Tespiti Sistemi") as demo:
    gr.Markdown(
        """
        # 🎯 YOLOv8 Derin Öğrenme Tabanlı Nesne Tespiti Sistemi
        **Model:** Ultralytics YOLOv8n | **Veri Seti:** COCO | **Görev:** Object Detection
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔍 Tekli Görüntü Tespiti"):
            with gr.Row():
                with gr.Column():
                    in_img = gr.Image(type="pil", label="Görüntü Yükle")
                    conf_s = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="Güven Eşiği (Confidence Threshold)")
                    iou_s = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU Eşiği")
                    btn_run = gr.Button("🚀 Nesne Tespiti Yap", variant="primary")
                with gr.Column():
                    out_img = gr.Image(type="pil", label="Model Çıktısı (Bounding Box)")
                    out_md = gr.Markdown("Görüntü yükleyip butona tıklayın.")
                    out_tbl = gr.Dataframe(
                        headers=["#", "Sınıf (Class)", "Güven Skoru", "Kutu [x1, y1, x2, y2]", "Tahmin Süresi"],
                        label="Tespit Edilen Nesneler",
                    )
            btn_run.click(tekli_tahmin, inputs=[in_img, conf_s, iou_s], outputs=[out_img, out_md, out_tbl])

        with gr.TabItem("📱 Çoklu Görüntü (5 Fotoğraf) Test ve Karşılaştırma"):
            gr.Markdown(
                "Test etmek istediğiniz **5 farklı fotoğrafı** yükleyin, gerçek nesne sınıflarını girin ve karşılaştırma tablosunu oluşturun."
            )
            with gr.Row():
                f1 = gr.Image(type="pil", label="Fotoğraf 1")
                f2 = gr.Image(type="pil", label="Fotoğraf 2")
                f3 = gr.Image(type="pil", label="Fotoğraf 3")
                f4 = gr.Image(type="pil", label="Fotoğraf 4")
                f5 = gr.Image(type="pil", label="Fotoğraf 5")
            with gr.Row():
                g1 = gr.Textbox(value="cup", label="Foto 1 Gerçek Nesne")
                g2 = gr.Textbox(value="bottle", label="Foto 2 Gerçek Nesne")
                g3 = gr.Textbox(value="laptop", label="Foto 3 Gerçek Nesne")
                g4 = gr.Textbox(value="chair", label="Foto 4 Gerçek Nesne")
                g5 = gr.Textbox(value="cell phone", label="Foto 5 Gerçek Nesne")

            b_conf = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="Güven Eşiği")
            btn_5 = gr.Button("📊 5 Fotoğrafı Değerlendir ve Karşılaştır", variant="primary")
            gal_out = gr.Gallery(label="Görsel Sonuçlar", columns=5, height=300)
            comp_md = gr.Markdown()

            btn_5.click(
                bes_fotograf_degerlendir,
                inputs=[f1, g1, f2, g2, f3, g3, f4, g4, f5, g5, b_conf],
                outputs=[gal_out, comp_md],
            )

        with gr.TabItem("📈 Model Performans Metrikleri"):
            gr.Markdown(metrikleri_oku())

if __name__ == "__main__":
    demo.launch(inbrowser=True)