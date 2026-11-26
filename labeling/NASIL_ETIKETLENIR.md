# Kontrol etiketlemesi nasıl yapılır? (1 sayfa)

**Amaç:** EFFIS sınırlarından bağımsız, ekibin kendi gözüyle verdiği "yandı / yanmadı" etiketleriyle tüm
yöntemleri (ve EFFIS'in kendisini) bir kez daha kontrol etmek. Bu iş **isteğe bağlıdır**; yapılmazsa makalede
"gelecek çalışma" olarak yazılır. Etiketler **asla kodla doldurulmaz**.

## Dosyalar
- `points.csv` — 300 nokta (her yangın için 150). Doldurulacak sütunlar: `label`, `confidence`, `comment`.
- `points.geojson` — aynı noktalar; QGIS veya Google Earth'te açılabilir.
- `points_key.csv` — noktaların hangi bölgeden (EFFIS içi / 0–1 km / 1–3 km) seçildiğini tutar.
  **Etiketleme bitene kadar bu dosyaya bakmayın** (kör etiketleme, önyargıyı önler).

## Adımlar (nokta başına ~30 saniye)
1. `points.csv` dosyasını Excel/LibreOffice ile açın (UTF-8, virgülle ayrılmış).
2. Satırdaki `copernicus_browser` bağlantısını açın (ücretsiz Copernicus Data Space hesabı gerekir).
3. Sol menüden **Sentinel-2 L2A** seçin. Tarihi önce `pre_date` (yangın öncesi), sonra `post_date`
   (yangın sonrası) yapın. Görselleştirme olarak **"False color (urban)"** veya **"SWIR"** kullanın:
   yanık alanlar koyu kahverengi/bordo, sağlıklı bitki örtüsü yeşil görünür.
4. Haritanın ortasındaki noktaya (20 m × 20 m'lik bir piksel) bakın:
   - Öncesinde bitki örtüsü var, sonrasında kararmış/kahverengi → `label = 1` (yandı)
   - İki tarihte de aynı (bitki örtüsü, tarla, yerleşim, su…) → `label = 0` (yanmadı)
5. `confidence`: emin iseniz `2`, kararsızsanız `1`. Nokta bulut/duman altındaysa `label` boş bırakın ve
   `comment` sütununa "bulut" yazın.
6. Mümkünse iki kişi ayrı ayrı etiketlesin, anlaşmazlıkları birlikte konuşup tek değere indirin.

## Sonra
Dosyayı kaydedin (sütun adlarını değiştirmeyin) ve şunu çalıştırın:

```
python scripts/75_eval_checklabels.py
```

Sonuçlar `outputs/tables/checklabels_eval.csv` dosyasına yazılır; ardından `python run_all.py` ile makale
sayıları yeniden üretilebilir.
