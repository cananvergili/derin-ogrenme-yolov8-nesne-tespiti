# Derin Öğrenme - YOLOv8 ile Nesne Tespiti (Object Detection) Uygulaması

Bu proje, **COCO** veri seti ve **Ultralytics YOLOv8n** derin öğrenme mimarisi kullanılarak geliştirilmiş bir Nesne Tespiti uygulamasıdır. Eğitilen model **Gradio** web arayüzü ile sunulmuştur.

## Dosya Yapısı
- 1_veri_hazirla.py: COCO veri setini indirir ve Eğitim (%70), Doğrulama (%15), Test (%15) olarak ayırır.
- 2_egitim_ve_degerlendirme.py: YOLOv8n modelini eğitir ve Test kümesi metriklerini (IoU, mAP@50, mAP@50-95, Precision, Recall, F1-Score, Inference Time) hesaplar.
- 3_uygulama_gradio.py: Eğitilen modeli Gradio web arayüzü olarak başlatır.
- 4_rapor_olustur.py: Proje raporunu (Derin_Ogrenme_Proje_Raporu.docx) otomatik oluşturur.
- Derin_Ogrenme_Proje_Raporu.docx: Proje raporu.
