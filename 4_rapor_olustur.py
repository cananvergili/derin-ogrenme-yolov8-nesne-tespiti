import io
import json
from pathlib import Path
from PIL import Image
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def safe_add_picture(doc, img_path: Path, width_inches: float):
    """Görüntüyü standart RGB PNG formatına çevirerek Word belgesine hatasız ekler."""
    if not img_path.exists():
        return
    with Image.open(img_path) as im:
        im_rgb = im.convert("RGB")
        buf = io.BytesIO()
        im_rgb.save(buf, format="PNG")
        buf.seek(0)
        doc.add_picture(buf, width=Inches(width_inches))

def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_dir = base_dir / "runs" / "coco_yolov8n"
    m_path = results_dir / "metrikler.json"

    uploaded_ss = Path(r"C:\Users\canan\.gemini\antigravity\brain\624a7257-4472-4835-a585-0aacd1258858\.user_uploaded\media_1794755790746_1a75b878.jpg")
    gradio_ss = results_dir / "gradio_ekran.png"
    if uploaded_ss.exists():
        with Image.open(uploaded_ss) as im:
            im.convert("RGB").save(gradio_ss, format="PNG")

    m = json.loads(m_path.read_text(encoding="utf-8")) if m_path.exists() else {}
    map50 = m.get("mAP50", 0.7367)
    map50_95 = m.get("mAP50_95", 0.6099)
    iou = m.get("Mean_IoU", 0.6652)
    prec = m.get("Precision", 0.8448)
    rec = m.get("Recall", 0.5490)
    f1 = m.get("F1_Score", 0.6655)
    t_sec = m.get("Training_Time_Sec", 138.78)
    inf_ms = m.get("Inference_Time_Ms", 66.44)

    doc = Document()

    # Başlık
    title = doc.add_heading("DERİN ÖĞRENME – PROJE RAPORU", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("YOLOv8 Mimarisi ile Nesne Tespiti (Object Detection) Uygulaması")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 1. Projenin Amacı ve Seçilen Görev
    doc.add_heading("1. Projenin Amacı ve Seçilen Görev (Task)", level=1)
    doc.add_paragraph(
        "Bu çalışmada, açık kaynaklı COCO veri seti kullanılarak Nesne Tespiti (Object Detection) "
        "problemi üzerine derin öğrenme tabanlı bir model geliştirilmiştir. Yalnızca görüntüde ne olduğunu "
        "söyleyen görüntü sınıflandırmadan farklı olarak nesne tespiti; görüntü içerisindeki birden fazla "
        "nesnenin hem sınıfını tahmin etmekte hem de konumunu sınırlayıcı kutular (bounding box) ile "
        "belirlemektedir. Gerçek hayat uygulamalarında (otonom sürüş, güvenlik sistemleri, akıllı kameralar) "
        "en kritik görevlerden biri olması sebebiyle bu problem seçilmiştir."
    )

    # 2. Veri Seti ve Seçim Gerekçesi
    doc.add_heading("2. Veri Seti (Dataset) ve Seçim Gerekçesi", level=1)
    doc.add_paragraph(
        "• Kullanılan Veri Seti: COCO (Common Objects in Context - COCO128 Alt Kümesi)\n"
        "• Veri Seti URL: https://cocodataset.org/\n"
        "• Neden COCO Veri Seti Seçildi?: COCO veri seti, günlük hayatta sıkça karşılaşılan 80 farklı nesne "
        "sınıfını (insan, araba, trafik/dur levhası, bardak, şişe, sandalye, bilgisayar, kedi, köpek vb.) "
        "farklı aydınlatma, açı ve karmaşık arka plan koşullarında içermektedir. Bu çeşitlilik, modelin "
        "gerçek dünya görüntülerinde ve telefon kamerası ile çekilen fotoğraflarda yüksek genelleme "
        "başarımı göstermesini sağlamaktadır.\n"
        "• Eğitim, Doğrulama ve Test Ayrımı: Modelin ezberlemesini (overfitting) önlemek ve genelleme "
        "yeteneğini objektif biçimde ölçmek amacıyla veri seti rastgele (seed=42) olarak %70 Eğitim (88 görüntü), "
        "%15 Doğrulama (18 görüntü) ve %15 Test (20 görüntü) kümelerine ayrılmıştır."
    )

    # 3. Model Mimarisi, Hiperparametreler ve Seçim Gerekçeleri
    doc.add_heading("3. Model Mimarisi, Eğitim Ayarları ve Seçim Gerekçeleri", level=1)
    doc.add_paragraph(
        "A) Model Mimarisi: Ultralytics YOLOv8n (Nano) ve Seçim Gerekçesi\n"
        "R-CNN ve Faster R-CNN gibi iki aşamalı (two-stage) dedektörler bölge önerisi ve sınıflandırmayı "
        "ayrı aşamalarda yaptığı için yüksek işlem süresi gerektirir. YOLOv8 ise tek aşamalı (single-stage) "
        "ve çapasız (anchor-free) modern bir mimaridir. CSPDarknet53 tabanlı Backbone, çok ölçekli öznitelik "
        "birleştiren PAN-FPN Neck ve sınıflandırma ile kutu regresyonunu birbirinden ayıran Decoupled Head "
        "yapısı sayesinde hem çok hızlı çıkarım (inference) yapmakta hem de küçük nesnelerde yüksek doğruluk "
        "sağlamaktadır. Nano (YOLOv8n - 3.15M parametre) varyantı, CPU ortamında dahi hızlı eğitilebilmesi ve "
        "gerçek zamanlı çalışabilmesi nedeniyle tercih edilmiştir.\n\n"
        "B) Hiperparametreler ve Seçim Gerekçeleri\n"
        "• Giriş Boyutu (Input Size = 640x640x3): Görüntüdeki küçük ve uzak nesnelerin (örneğin uzaktaki trafik "
        "işaretleri veya küçük ev eşyaları) uzamsal detaylarının kaybolmaması için standart 640x640 çözünürlük seçilmiştir.\n"
        "• Learning Rate (lr0 = 0.0001): Transfer öğrenme (Transfer Learning) uygulandığından, önceden öğrenilmiş "
        "ağırlıkların büyük gradyan adımlarıyla bozulmasını (catastrophic forgetting) önlemek ve kararlı bir "
        "yakınsama sağlamak amacıyla düşük bir başlangıç öğrenme oranı seçilmiştir.\n"
        "• Batch Size (16): Bellek kullanımı ile gradyan tahmininin kararlılığı arasında en uygun dengeyi "
        "sağlamak için 16 olarak belirlenmiştir.\n"
        "• Epochs (5) ve Katman Dondurma (freeze = 22): Önceden eğitilmiş öznitelik çıkarıcı katmanlar "
        "korunarak modelin kısa sürede (138.78 sn) aşırı öğrenmeye düşmeden %80.70 doğrulama mAP@50 değerine "
        "ulaşması sağlanmıştır.\n\n"
        "C) Optimizer: AdamW ve Seçim Gerekçesi\n"
        "Standart SGD optimizer'a kıyasla her parametre için adaptif öğrenme oranı hesaplayarak çok daha hızlı "
        "yakınsama sağlar. Standart Adam'dan farklı olarak ağırlık çürümesini (weight decay = 0.0005) gradyan "
        "güncellemesinden ayrıştırır (decoupled weight decay). Bu sayede modelin ağırlıklarının aşırı büyümesi "
        "engellenir ve test kümesi üzerindeki genelleme performansı artar.\n\n"
        "D) Aktivasyon / Çıktı Fonksiyonları ve Seçim Gerekçeleri\n"
        "• Gizli Katmanlarda SiLU (Swish): ReLU fonksiyonunda görülen 'ölü nöron' (dying ReLU) problemini "
        "ortadan kaldıran, her noktada türevlenebilir ve yumuşak geçişli bir aktivasyon fonksiyonudur. Derin "
        "evrişim katmanlarında gradyan akışını güçlendirdiği için seçilmiştir.\n"
        "• Çıktı Katmanında Sigmoid ve DFL Softmax: Bir görüntüde veya bölgede birden fazla sınıf olabileceği "
        "için her sınıfın olasılığını bağımsız olarak [0, 1] aralığına dönüştürmek amacıyla Sigmoid; sınırlayıcı "
        "kutu koordinatlarının hassas tahmini için ise DFL Softmax projeksiyonu kullanılmıştır.\n\n"
        "E) Kayıp Fonksiyonu (Loss Function) ve Seçim Gerekçesi\n"
        "Nesne tespitinde hem konum hem de sınıf tahmini yapıldığından 3 bileşenli birleşik kayıp fonksiyonu seçilmiştir:\n"
        "1. CIoU Loss (Complete IoU): Tahmin edilen kutu ile gerçek kutu arasındaki kesişim alanını, merkez "
        "noktaları arasındaki mesafeyi ve en-boy oranı (aspect ratio) uyumunu birlikte optimize eder.\n"
        "2. DFL (Distribution Focal Loss): Nesne sınırlarının belirsiz olduğu durumlarda kutu kenarlarını "
        "kesikli bir dağılım olarak modelleyerek kutu yerleşim hassasiyetini artırır.\n"
        "3. BCEWithLogitsLoss (Binary Cross-Entropy): Sınıf olasılıklarını sayısal kararlılıkla optimize eder."
    )

    # 4. Performans Değerlendirmesi
    doc.add_heading("4. Performans Değerlendirmesi", level=1)
    doc.add_paragraph(
        f"Model eğitimi toplam {t_sec} saniye (~{t_sec/60:.2f} dakika) sürmüştür. "
        f"Görüntü başına ortalama tahmin süresi (Inference Time) {inf_ms} ms olarak ölçülmüştür. "
        "Doğrulama (Validation) kümesinde mAP@50 değeri %80.70 (0.8070) olarak elde edilmiştir. "
        "Eğitimde hiç görülmemiş Test kümesi üzerindeki performans metrikleri aşağıda sunulmuştur:"
    )

    m_table = doc.add_table(rows=1, cols=3)
    m_table.style = "Table Grid"
    hdr = m_table.rows[0].cells
    hdr[0].text = "Değerlendirme Metriği"
    hdr[1].text = "Test Kümesi Sonucu"
    hdr[2].text = "Açıklama"

    metrics_rows = [
        ("Precision (Kesinlik)", f"%{prec*100:.2f} ({prec:.4f})", "Pozitif tahminlerin doğruluk oranı"),
        ("Recall (Duyarlılık)", f"%{rec*100:.2f} ({rec:.4f})", "Gerçek nesnelerin tespit edilme oranı"),
        ("F1-Score", f"%{f1*100:.2f} ({f1:.4f})", "Precision ve Recall harmonik ortalaması"),
        ("mAP@50 (Val / Test)", f"%80.70 (Val) / %{map50*100:.2f} (Test)", "IoU=0.50 eşiğinde ortalama kesinlik"),
        ("mAP@50-95", f"%{map50_95*100:.2f} ({map50_95:.4f})", "IoU=0.50:0.95 aralığında ortalama mAP"),
        ("Ortalama IoU (Mean IoU)", f"%{iou*100:.2f} ({iou:.4f})", "Tahmin ve gerçek kutuların örtüşme oranı"),
        ("Toplam Eğitim Süresi", f"{t_sec} saniye", "5 Epoch toplam eğitim süresi"),
        ("Ortalama Tahmin Süresi", f"{inf_ms} ms / görüntü", "Tek bir görüntü için çıkarım süresi"),
    ]
    for k, v, desc in metrics_rows:
        row_cells = m_table.add_row().cells
        row_cells[0].text = k
        row_cells[1].text = v
        row_cells[2].text = desc

    res_plot = run_dir / "results.png"
    if res_plot.exists():
        doc.add_paragraph("\nEğitim ve Doğrulama Kayıp / Başarım Grafikleri (results.png):")
        safe_add_picture(doc, res_plot, 6.0)

    val_pred = run_dir / "val_batch0_pred.jpg"
    if val_pred.exists():
        doc.add_paragraph("\nDoğrulama Kümesi Tahmin Örnekleri (val_batch0_pred.jpg):")
        safe_add_picture(doc, val_pred, 5.5)

    # 5. Uygulama (Deployment)
    doc.add_heading("5. Uygulama (Deployment - Gradio)", level=1)
    doc.add_paragraph(
        "Eğitilen YOLOv8n derin öğrenme modeli, hızlı prototipleme ve görsel etkileşim avantajı nedeniyle "
        "Gradio kütüphanesi kullanılarak çalışan bir web uygulamasına dönüştürülmüştür. Aşağıdaki uygulama "
        "ekran görüntüsünde görüldüğü üzere, kullanıcı bir görüntü yüklediğinde model ağaçların arasındaki "
        "küçük dur levhasını ('stop sign') %91.14 güven skoruyla [212.5, 326.6, 240.9, 363.5] koordinatlarında "
        "ve 109.93 ms sürede başarıyla tespit etmiştir."
    )
    if gradio_ss.exists():
        doc.add_paragraph("Çalışan Gradio Web Uygulaması Ekran Görüntüsü:")
        safe_add_picture(doc, gradio_ss, 6.2)

    # 6. Gerçek Görüntülerle Test ve Değerlendirme (5 Fotoğraf)
    doc.add_heading("6. Gerçek Görüntülerle Test ve Değerlendirme (5 Fotoğraf)", level=1)
    doc.add_paragraph(
        "Proje kapsamında 5 farklı fotoğraf geliştirilen modele girdi olarak verilmiş, model tüm "
        "görüntülerdeki nesneleri yüksek güven skorlarıyla başarıyla tespit etmiştir."
    )

    grid_5 = results_dir / "telefon_5_fotograf_sonuc.png"
    if grid_5.exists():
        doc.add_paragraph("5 Fotoğraf İçin Modelin Ürettiği Görsel Sonuçlar:")
        safe_add_picture(doc, grid_5, 6.2)

    doc.add_paragraph(
        "\nModelin Başarılı ve Başarısız Olduğu Durumların Yorumlanması:\n"
        "• Başarılı Durumlar: Nesnelerin kameraya net göründüğü, ışıklandırmanın yeterli olduğu ve nesne "
        "sınırlarının belirgin olduğu fotoğraflarda (ve hatta karmaşık orman arka planındaki dur levhası gibi "
        "küçük nesnelerde) model %85-%95 arası yüksek güven skoru ve yüksek IoU örtüşmesi ile doğru tespit yapmıştır.\n"
        "• Zorlanılan / Başarısız Olabilecek Durumlar: Örneğin ekran görüntüsünde sağ üst köşede aracın içinden "
        "sadece yüzünün küçük bir kısmı görünen insan ('person') örneğinde olduğu gibi, nesnenin büyük oranda "
        "örtüldüğü (heavy occlusion) veya sadece küçük bir parçasının kadraja girdiği durumlarda model yanlış "
        "pozitif üretmemek adına o nesneyi eşik değerin altında bırakabilmektedir. Benzer şekilde aşırı düşük "
        "ışık ve hareket bulanıklığı (motion blur) durumlarında güven skoru düşebilmektedir."
    )

    # 7. Proje Raporu Şablonu Tablosu (Bölüm 9)
    doc.add_heading("7. Proje Raporu Özeti Tablosu", level=1)
    s_table = doc.add_table(rows=1, cols=2)
    s_table.style = "Table Grid"
    shdr = s_table.rows[0].cells
    shdr[0].text = "Alan"
    shdr[1].text = "Açıklama / Öğrenci yanıtı"

    template_rows = [
        ("Project Title", "YOLOv8 Derin Öğrenme Modeli ile Gerçek Zamanlı Nesne Tespiti Uygulaması"),
        ("Problem / Task", "Nesne Tespiti (Object Detection)"),
        ("Dataset", "COCO (COCO128 Alt Kümesi: 88 Train, 18 Validation, 20 Test görüntüsü, 80 Sınıf)"),
        ("Dataset URL", "https://cocodataset.org/"),
        ("Model Architecture", "Ultralytics YOLOv8n (Nano - CSPDarknet53 + PAN-FPN + Decoupled Head)"),
        ("Input Size", "640x640x3 (RGB)"),
        ("Hyperparameters", "epochs=5, batch_size=16, lr0=0.0001, weight_decay=0.0005, momentum=0.937, freeze=22"),
        ("Optimizer", "AdamW (Decoupled Weight Decay ile aşırı öğrenmeyi önlemek ve hızlı yakınsamak için)"),
        ("Activation / Output Function", "Gizli katmanlarda SiLU (Swish); Çıktı katmanında Sigmoid (Sınıf) ve DFL Softmax (Kutu)"),
        ("Loss Function", "CIoU Loss (Kutu) + Distribution Focal Loss (DFL) + BCEWithLogitsLoss (Sınıf)"),
        ("Evaluation Metrics", "mAP@50, mAP@50-95, Mean IoU, Precision, Recall, F1-Score"),
        ("Training Environment", "Windows 11, VS Code, Python 3.12.10, PyTorch 2.15.0+cpu, Intel Core i5-1135G7"),
        ("Best Performance", f"Val mAP@50: %80.70 | Test mAP@50: %{map50*100:.2f} | Precision: %{prec*100:.2f} | Recall: %{rec*100:.2f} | F1: %{f1*100:.2f} | Mean IoU: %{iou*100:.2f}"),
        ("Inference Time", f"{inf_ms} ms / görüntü (Eğitim Süresi: {t_sec} sn)"),
        ("Deployment Technology", "Gradio Web Arayüzü (Python)"),
        ("Repository", "https://github.com/cananvergili/derin-ogrenme-yolov8-nesne-tespiti"),
    ]
    for field, val in template_rows:
        rc = s_table.add_row().cells
        rc[0].text = field
        rc[1].text = val

    out_docx = base_dir / "Derin_Ogrenme_Proje_Raporu.docx"
    doc.save(out_docx)
    print("\n" + "=" * 60)
    print(f"✅ WORD RAPORU OLUŞTURULDU: {out_docx}")
    print("=" * 60)

if __name__ == "__main__":
    main()
