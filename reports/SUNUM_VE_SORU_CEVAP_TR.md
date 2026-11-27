# Sunum ve Soru-Cevap Rehberi (TR + EN)

Final sunumu (28 Aralık 2026, 12 dk + 3 dk demo + 5 dk soru) ve sözlü kontrol (ders bölüm 10: "Hoca herhangi bir
üyeye kodun veya makalenin herhangi bir kısmını sorabilir") için hazırlık. Kısa cevaplar önce Türkçe, sonra
İngilizce. Sayılar için en alttaki otomatik bloğa bakın (elle yazılmadı).

---

## 1. Olası sorular

**S1. Neden Yunanistan'dan Türkiye'ye aktarım? / Why transfer from Greece?**
TR: Türkiye için uzman etiketli, açık, makine öğrenmesine hazır bir yanık veri seti yok; Yunanistan'daki FLOGA
326 yangın içeriyor ve iklim/bitki örtüsü (Akdeniz çam ormanı, maki) Türkiye'nin güneybatısına çok benziyor. Türkiye'de
etiket toplamadan çalışan bir model, yeni bir yangında hemen kullanılabilir.
EN: There is no open, expert-labelled ML-ready burnt-area dataset for Türkiye; FLOGA has 326 Greek fires in a very
similar Mediterranean setting. A model that needs no Turkish labels can be used immediately after a new fire.

**S2. Neden EFFIS referans? / Why EFFIS as the reference?**
TR: Avrupa Komisyonu'nun resmî, herkese açık, iki yangını da kapsayan yanık alan sınırları. Alternatif (OGM saha
ölçümleri) açık değil. EFFIS'in de Sentinel-2'den üretildiğini makalede sınırlılık olarak yazdık ve bağımsız kontrol
için kör etiketleme kiti hazırladık.
EN: It is the official, open EU product covering both fires; field data of OGM are not public. We state that EFFIS
is also Sentinel-2-based, and provide a blind check-labelling kit.

**S3. Referans bağımsız mı? / Is the reference independent?**
TR: Yöntem olarak bağımsız (EFFIS analistlerin yarı otomatik çizimi; bizim modeller farklı algoritmalar), veri
olarak değil (aynı uydu). Bu nedenle sonuçlar "EFFIS ile uyum" olarak yorumlanmalı; bunu makalede açıkça yazdık.
EN: Independent in method, not in sensor; we report agreement with EFFIS and say so explicitly.

**S4. Neden bu eşikler (0.10, 0.27)? / Why these thresholds?**
TR: USGS FIREMON'un (Key & Benson 2006) yayımlanmış düşük ve orta şiddet sınırları; Türkiye verisine bakılarak
seçilmedi. Üçüncü dNBR varyantının eşiği yalnızca FLOGA doğrulama verisinde F1'i en yükseğe çıkaran değerdir.
EN: Published USGS severity limits, fixed a priori; the third variant is tuned on FLOGA validation only.

**S5. Veri sızıntısını (leakage) nasıl önlediniz? / How did you prevent leakage?**
TR: (1) Türkiye verisi eğitim, eşik, hiperparametre, özellik seçimi veya erken durdurmada hiç kullanılmadı;
(2) `tests/test_no_leakage.py` hazırlık/ayar/eğitim betiklerinin Türkiye verisini içe aktarmadığını otomatik
denetliyor; (3) FLOGA'nın resmî olay bazlı bölmesi kullanıldı, bir yangının pikselleri iki bölmeye düşmüyor.
EN: Turkish data enter only evaluation scripts; a unit test enforces it; FLOGA's event-level split is used.

**S6. Neden LightGBM? / Why LightGBM?**
TR: Tablo verisinde (piksel × özellik) çok güçlü, hızlı, GPU gerektirmiyor, model dosyası küçük; Random Forest'a
göre daha az bellekle benzer/daha iyi doğruluk. FLOGA makalesindeki XGBoost'un daha hızlı bir akrabası.
EN: Strong on tabular per-pixel features, fast, CPU-only, tiny model files.

**S7. Maliyeti nasıl ölçtünüz? / How did you measure cost?**
TR: Aynı dizüstü bilgisayarda (Ryzen 7 6800H, 16 GB, RTX 3050 Laptop GPU): parametre sayısı, model boyutu, eğitim süresi,
1000 km² başına CPU ve GPU çıkarım süresi, en yüksek RAM/VRAM (`src/burnt/cost.py`, `outputs/tables/cost_table.csv`).
BAM-CD'nin eğitim maliyetini ölçmedik (önceden eğitilmiş ağırlık kullandık); FLOGA yapılandırmasına göre onun eğitimi GPU üzerinde 150 epoch sürer (`external/FLOGA/configs/config.json`).
EN: Same laptop; params, size, train time, CPU/GPU inference time per 1000 km², peak memory.

**S8. BAM-CD'den kötüyse ne olacak? / What if you are worse than BAM-CD?**
TR: Dürüstçe raporluyoruz (ders 5.6). Katkımız "aynı doğruluk" değil, "çok daha düşük maliyetle yakın doğruluk" ve
"ilk nicel FLOGA→Türkiye aktarım testi". Farkın güven aralığını veriyoruz; hangi arazi örtülerinde kaybettiğimizi
hata analizinde gösteriyoruz.
EN: We report it honestly; our claim is accuracy close to BAM-CD at much lower cost plus the first quantitative
transfer test, with CIs and an error analysis.

**S9. Yapay zekâ kullanımı? / AI use?**
TR: Kod üretimi, deney düzeni ve metin taslağı için Claude Code (Anthropic) kullanıldı; ara dönemde Claude ve Gemini
kod yardımı ve dil düzeltmesi için kullanıldı. Bütün kodu ve metni okuduk, sonuçlardan biz sorumluyuz. Bu, makalede
ve README'de yazıyor.
EN: Claude Code for code generation, experiment orchestration and drafting; authors reviewed everything.

**S10. FLOGA'yı nasıl indirdiniz? 1.1 TB değil mi? / How did you get FLOGA?**
TR: İndirmedik. Hugging Face'teki tar dosyalarının başlıklarını HTTP "range" istekleriyle okuyup her GeoTIFF'in
dosya içindeki konumunu çıkardık (`scripts/hf_tar_index.py`); GeoTIFF'ler sıkıştırılmamış olduğu için sadece gereken
256×256 yamaların baytlarını okuduk (`src/burnt/remote_tiff.py`).
EN: We indexed the tar headers by HTTP range requests and read only the bytes of the needed patches.

**S11. FLOGA v1 ile v2 farkı? / v1 vs v2?**
TR: v2, Copernicus'un yeniden işlediği görüntüler (işleme sürümü N0500, yansıtıma +1000 ofset eklenmiş). Ofseti ve
B01 bandını çıkararak v1/BAM-CD ve 2021 Türkiye görüntüleriyle aynı birime getirdik; bant histogramlarını
karşılaştırdık (`outputs/figures/band_harmonisation.png`).

**S12. Neden 20 m? / Why 20 m?**
TR: FLOGA ve BAM-CD 20 m'de; kırmızı-kenar ve SWIR bantlarının doğal çözünürlüğü 20 m. 10 m bantlar ortalama alınarak
20 m'ye indirildi.

**S13. Blok bootstrap neden 1 km? / Why 1 km blocks?**
TR: Komşu pikseller ilişkili; 1 km (50×50 piksel) blok, yanık lekelerinin tipik genişliğinden büyük ve yangın alanında
yeterince çok blok bırakıyor. Ayrıntı: Künsch (1989).

**S14. Ablasyon ne gösterdi? / What did the ablation show?** → otomatik bloktaki ablasyon satırları.

**S15. Model nerede hata yapıyor? / Where does it fail?** → `outputs/figures/errors_by_landcover.png`,
`outputs/tables/errors_by_severity.csv`; yorum makalenin Discussion bölümünde.

## 2. Kod turu (her üye her dosyayı anlatabilmeli)

| Dosya | 1–2 cümlelik açıklama |
|---|---|
| `configs/default.yaml` | Bütün sabitler: EFFIS kimlikleri, tarih pencereleri, bantlar, SCL sınıfları, örnekleme N=2000, bootstrap ayarları. |
| `src/burnt/stac.py` | STAC araması; aynı gün+aynı uydu tile'larını tek geçiş sayar, AOI'yi kaplayan ve ≥%97 bulutsuz ilk geçişi seçer, tile'ları mozaikler. |
| `src/burnt/preprocess.py` | `read_dn`: mozaik + işleme sürümü ≥4.0 ise −1000 ofset + 20 m ızgaraya yeniden örnekleme; `floga_scale`: min(DN,10000)/10000. |
| `src/burnt/reference.py` | EFFIS GeoJSON'u okur, AOI kutusunu (poligon + 3 km) çıkarır, iki tarih arasında başlayan yangınları rasterleştirir. |
| `scripts/10_build_turkey_testset.py` | İki olay için 9 bantlık önce/sonra yığınları, SCL, geçerli maske, EFFIS maskesini `data/cache/turkey/*.npz` olarak kaydeder. |
| `scripts/hf_tar_index.py`, `src/burnt/remote_tiff.py` | Tar başlıklarını 512 baytlık range istekleriyle okur; sıkıştırılmamış TIFF'in satır ofsetlerinden pencere okur; HF yönlendirmesini bir kez çözüp CDN adresini kullanır (429 hatasını önler). |
| `src/burnt/floga.py` | Resmî bölmeyi (yıl, satır no) → (yangın ID, tile) → v2 dosya adı olarak eşler; FLOGA'nın dolgu/yama ızgarasını birebir uygular; v2→v1 bant/ofset dönüşümü. |
| `scripts/20_prepare_floga.py` | A: maskeler ve yama tablosu; B: bulutlu/denizli/boş yamaları atıp 1:1 yama seçimi; C: görüntü pencerelerini okuma; D: yama başına 2000 piksel örnekleme. |
| `src/burnt/indices.py`, `features.py` | NBR, NBR2, NDVI, NDWI, BAIS2, MIRBI; fark indeksleri; G1/G2/G3/G4/ALL özellik grupları. |
| `scripts/30_tune_dnbr.py` | FLOGA doğrulamasında −0.20…1.00 arası 0.01 adımla en iyi F1 veren dNBR eşiğini bulur. |
| `scripts/40_hpo.py` | Optuna TPE + MedianPruner; amaç: FLOGA doğrulama piksellerinde en iyi eşikteki F1. |
| `scripts/50_train_final.py` | En iyi ayarlarla 5 tohum; her tohum için eşik doğrulamada seçilir; FLOGA test tahminleri, maliyet. |
| `src/burnt/bamcd.py`, `scripts/60_run_bamcd.py` | BAM-CD'nin resmî modelini yükler, FLOGA ölçeklemesiyle 256×256 kayan pencerede çalıştırır; FLOGA test kontrolü, Türkiye çıkarımı, süre/bellek. |
| `scripts/65_predict_turkey.py` | Eğitilmiş modelleri dondurulmuş eşiklerle Türkiye'ye uygular (sadece çıkarım). |
| `src/burnt/metrics.py`, `scripts/70_evaluate.py` | P/R/F1/IoU; 1 km blok bootstrap (eşleştirilmiş çekilişler), ΔF1 güven aralıkları; FLOGA test ve Türkiye tabloları. |
| `scripts/72_cost_table.py` | Parametre, boyut, eğitim ve çıkarım süreleri, bellek, doğrulukla birleşik tablo. |
| `scripts/80_ablation.py` | G1, G2, G2+G3, ALL özellik gruplarıyla LightGBM. |
| `scripts/90_error_analysis.py` | ESA WorldCover sınıflarına ve dNBR şiddet bantlarına göre FP/FN; anlaşmazlık haritaları. |
| `scripts/85_figures.py`, `95_make_paper_numbers.py` | Şekiller; makale/README/rapor sayılarını tablolardan otomatik üretir. |
| `tests/test_no_leakage.py` | Ayar/eğitim betiklerinin Türkiye verisine dokunmadığını denetler. |

## 3. Otomatik sayılar
<!-- KEYNUMBERS:START -->
- FLOGA test F1: BAM-CD v1 0.909 (makale 0.8848), LightGBM 0.864, RF 0.865, dNBR (FLOGA eşiği) 0.661
- Türkiye birleşik F1: LightGBM 0.878 [0.871, 0.887], BAM-CD v1 0.912, BAM-CD v2 0.925, dNBR 0.10 0.889, dNBR FLOGA eşiği 0.824
- Manavgat F1: LightGBM 0.882, BAM-CD v1 0.921; Marmaris F1: LightGBM 0.855, BAM-CD v1 0.866
- ΔF1 (LightGBM − BAM-CD v1), birleşik: -0.033 [-0.040, -0.027]; (LightGBM − dNBR FLOGA): +0.055
- Maliyet: LightGBM 35 834 parametre (ağaç düğümü), 1.9 MB, CPU 2.9 s/1000 km²; BAM-CD v1 83 727 022 parametre, CPU 50.2 s, GPU 5.4 s/1000 km²
- HPO: LightGBM 60 deneme, doğrulama F1 0.845; RF 14 deneme (2 saat sınırı), F1 0.846; dNBR eşiği 0.24
- Ablasyon (FLOGA doğrulama F1): G1 0.678, G2 0.839, G2+G3 0.841, ALL 0.844
- Veri: FLOGA 327/338 olay, yama train/val/test 538/184/162, eğitim örneği 710 625 piksel
<!-- KEYNUMBERS:END -->
