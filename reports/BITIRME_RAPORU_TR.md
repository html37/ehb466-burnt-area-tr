# EHB 466E Dönem Projesi — Türkçe Bitirme Raporu

**Konu:** 2021 Manavgat ve Marmaris orman yangınlarında yanmış alanların Sentinel-2 uydu görüntüleriyle
haritalanması; Yunanistan verisiyle (FLOGA) eğitilmiş hafif bir modelin Türkiye'ye aktarılması.
**Ekip:** Hüseyin Temel (040220042), Erdem Akkoca (040220057).

> Bu rapor, uzaktan algılama uzmanı olmayan bir okuyucu için yazıldı. Önce kavramlar sıfırdan anlatılıyor,
> sonra her aşamada ne yapıldığı ve neden yapıldığı, en sonda sonuçlar ve sınırlılıklar var. Rapordaki bütün
> sayılar `outputs/` klasöründeki dosyalardan `scripts/95_make_paper_numbers.py` tarafından otomatik
> eklenmiştir (elle yazılmış sonuç yoktur).

---

## 1. Proje ne yapıyor? (bir paragrafta)

Temmuz–Ağustos 2021'de Manavgat ve Marmaris'te büyük orman yangınları çıktı. Orman Genel Müdürlüğü (OGM) ve AFAD,
hasar tespiti ve yeniden ağaçlandırma için "tam olarak nerenin yandığını" gösteren haritalara ihtiyaç duyar. Bu
projede yangından **önce** ve **sonra** çekilmiş iki Sentinel-2 uydu görüntüsünü karşılaştırarak her 20 m × 20 m'lik
pikselin "yandı / yanmadı" olduğuna karar veren üç tür yöntemi aynı koşullarda yarıştırdık:

1. **dNBR eşiklemesi** — literatürdeki en basit ve en yaygın yöntem (benchmark 1),
2. **BAM-CD** — Yunan araştırmacıların 2024'te yayımladığı, 84 milyon parametreli derin öğrenme modeli (benchmark 2),
3. **Bizim yöntemimiz** — yalnızca Yunanistan verisiyle eğitilmiş, sadece CPU ile çalışan hafif bir LightGBM
   sınıflandırıcısı. **Türkiye'den tek bir piksel bile eğitimde kullanılmadı.**

Doğruluk, Avrupa Orman Yangını Bilgi Sistemi'nin (EFFIS) yangın sınırlarıyla ölçüldü.

## 2. Temel kavramlar (sıfırdan)

### 2.1 Uydu görüntüsü, bant ve yansıtım
Sentinel-2, Avrupa Uzay Ajansı'nın iki uydusundan (2A, 2B) oluşan bir görevdir; dünyanın her yerini yaklaşık 5 günde
bir görüntüler. Kamera, ışığı farklı dalga boylarında ayrı ayrı ölçer; her birine **bant** denir. Örneğin B04
kırmızı ışık, B8A yakın kızılötesi (NIR), B11 ve B12 kısa dalga kızılötesi (SWIR) bantlarıdır. Bir pikseldeki değer,
yüzeyin o dalga boyundaki ışığın ne kadarını geri yansıttığını (**yansıtım**, 0–1 arası) gösterir. Dosyalarda bu
değer 10 000 ile çarpılmış tam sayı olarak saklanır (DN).

### 2.2 Yanık alan neden görülebilir? NBR ve dNBR
Sağlıklı bitki yakın kızılötesini (NIR) çok yansıtır, kısa dalga kızılötesini (SWIR) az yansıtır. Yanmış alanda
tersi olur: NIR düşer, SWIR artar. Bu yüzden **NBR = (NIR − SWIR2) / (NIR + SWIR2)** sağlıklı bitkide yüksek,
yanıkta düşüktür. Yangından önceki NBR'den sonrakini çıkarırsak **dNBR** elde ederiz; büyük dNBR "burada bir şey
yandı" demektir. ABD Jeoloji Araştırmaları Kurumu (USGS) dNBR ≥ 0.10'u "düşük şiddetli yanık", ≥ 0.27'yi "orta"
olarak sınıflar; bu iki sayı bizim sabit eşiklerimizdir. Benzer şekilde NDVI, NDWI, NBR2, BAIS2 ve MIRBI gibi başka
**spektral indeksler** de vardır (formüller `src/burnt/indices.py`, FLOGA makalesi Tablo IV).

### 2.3 Maske (bulut, gölge, su)
Bulutlu veya gölgeli piksellerde yüzey görünmez; deniz/su ise yanık gibi karanlık görünebilir. Sentinel-2 L2A
ürünü her piksel için bir **sahne sınıfı (SCL)** verir (bulut, gölge, su, bitki…). Bu sınıflara bakarak güvenilmez
pikselleri **maskeliyoruz**, yani değerlendirme dışında bırakıyoruz. Bütün yöntemler aynı maskeyi kullanır.

### 2.4 Mozaik ve tile
Sentinel-2 görüntüleri 110 km × 110 km'lik karelere (**tile**) bölünür. Manavgat yangını iki tile'ın sınırına
düşüyor. Aynı uydu geçişine ait iki tile'ı birleştirip tek bir görüntü yapmaya **mozaik** denir. Böylece yangın
alanının tamamı kapsandı.

### 2.5 Eğitim / doğrulama / test ayrımı
Makine öğrenmesi modeli örneklerden öğrenir. Dürüst bir değerlendirme için veri üçe ayrılır:
- **Eğitim (train):** model bununla öğrenir.
- **Doğrulama (validation, val):** ayarlar (eşik, hiperparametre) bununla seçilir.
- **Test:** sadece en sonda, bir kez, başarıyı ölçmek için kullanılır.
Bizim projede eğitim ve doğrulama Yunanistan'dan (FLOGA), test ise Manavgat ve Marmaris'tir. Test verisine bakarak
herhangi bir ayar yapmak "kopya çekmek" olurdu; bunu bir otomatik test (`tests/test_no_leakage.py`) engelliyor.
FLOGA'nın kendi resmî bölmesi **yangın olayı bazında** yapılmıştır: bir yangının pikselleri hem eğitimde hem testte
yer almaz.

### 2.6 Hiperparametre optimizasyonu (HPO)
Modelin kendi kendine öğrenmediği ayarlarına (ağaç sayısı, öğrenme hızı, yaprak sayısı…) **hiperparametre** denir.
Optuna kütüphanesi bu ayarları akıllıca deneyerek doğrulama verisinde en iyi sonucu vereni bulur (TPE algoritması;
umutsuz denemeleri erkenden kesen "pruner"). Biz her model için en fazla 60 deneme / 2 saat ayırdık.

### 2.7 Metrikler: Precision, Recall, F1, IoU
- **Precision (kesinlik):** "Yandı" dediğimiz piksellerin ne kadarı gerçekten yanmış?
- **Recall (duyarlılık):** Gerçekten yanan piksellerin ne kadarını bulabildik?
- **F1:** ikisinin harmonik ortalaması; tek sayıyla dengeli başarı. Ana ölçütümüz budur.
- **IoU:** tahmin ile referansın kesişiminin birleşimlerine oranı (F1'e benzer, daha serttir).
Hata türleri: **FP (yanlış alarm)** yanmamış yere "yandı" demek; **FN (kaçırma)** yanmış yeri bulamamak.

### 2.8 Belirsizlik: 5 tohum ve blok bootstrap
- Makine öğrenmesi modelleri rastgelelik içerir. Her modeli 5 farklı rastgele **tohum (seed)** ile eğitip
  ortalama ± standart sapma veriyoruz.
- Haritadaki komşu pikseller birbirine çok benzer; bu yüzden pikselleri bağımsız sayarak güven aralığı hesaplamak
  yanıltıcı olur. **Uzamsal blok bootstrap**'te harita 1 km × 1 km bloklara bölünür, bloklar 1000 kez rastgele
  (iadeli) yeniden seçilir ve her seferinde F1 hesaplanır. Bu 1000 değerin %2.5 ve %97.5'lik dilimleri **%95 güven
  aralığını** verir. Aynı çekilişler bütün yöntemlere uygulanır; böylece "bizim F1 − BAM-CD F1" farkının güven
  aralığı da hesaplanır. Aralık 0'ı içermiyorsa fark istatistiksel olarak anlamlıdır.

### 2.9 Ablasyon
Modelin hangi parçasının işe yaradığını görmek için parçaları tek tek çıkarıp yeniden deneriz. Biz özellik
gruplarını karşılaştırdık: G1 (sadece dNBR), G2 (6 fark indeksi), G2+G3 (+ yangın sonrası bantlar), ALL (36 özellik).

### 2.10 BAM-CD
FLOGA makalesindeki derin öğrenme modeli. İki görüntüyü (önce/sonra) aynı ResNet-101 kodlayıcıdan geçirir (Siyam
ağ), özellikleri birleştirip U-Net tarzı bir kod çözücüyle her piksel için yanık olasılığı üretir. 83.7 milyon
parametresi vardır ve GPU ile eğitilmiştir. Biz onu yeniden eğitmedik: yazarların resmî kodunu ve eğitilmiş
ağırlıklarını olduğu gibi kullandık.

### 2.11 Aktarım (transfer) ve alan kayması
Bir ülkede eğitilen modelin başka bir ülkede kullanılmasına **aktarım** denir. Bitki türleri, toprak, mevsim ve
görüntü koşulları farklı olduğu için başarı düşebilir (**alan kayması**, domain shift). Projemizin ana sorusu:
Yunanistan verisiyle eğitilen modeller Türkiye'de ne kadar iyi çalışıyor?

### 2.12 LightGBM ve Random Forest
Her ikisi de çok sayıda **karar ağacını** birleştirir. Random Forest ağaçları bağımsız eğitip oylatır; LightGBM ise
ağaçları sırayla, bir öncekinin hatalarını düzeltecek şekilde ekler (gradyan artırma). Her piksel ayrı ayrı (36
sayılık bir özellik vektörüyle) sınıflandırılır; komşu piksellere bakılmaz. Çok hızlıdır, GPU gerektirmez.

## 3. Aşama aşama ne yaptık, neden?

| Aşama | Ne yapıldı | Neden |
|---|---|---|
| 0 Kurulum | Python 3.11 sanal ortam, paket sürümleri sabitlendi (`requirements-lock.txt`), donanım kaydı (`docs/hardware.md`), ders gereksinimleri listesi (`docs/requirements_checklist.md`) | Tekrarlanabilirlik, notlandırma kriterlerini eksiksiz karşılamak |
| 1 Yeniden düzenleme | Ara sunum notebook'undaki kod `src/burnt/` modüllerine taşındı; birim testleri yazıldı; dNBR tablosu birebir yeniden üretildi | Tek bir kod tabanıyla bütün yöntemleri aynı boru hattında çalıştırmak |
| 2 FLOGA | FLOGA'nın 1.1 TB'lık Hugging Face sürümü indirilmeden, tar dosyalarının içindeki GeoTIFF'lerin yalnızca gereken baytları HTTP "range" istekleriyle okundu; resmî olay bölmesi v2 dosyalarıyla eşleştirildi; FLOGA'nın yama kuralları (256×256, bulutlu/denizli/boş yamaların atılması, 1:1 yanık/yanmamış yama) uygulandı; piksel örneklemesi (yama başına 2000 piksel) yapıldı | Diskte 94 GB yer vardı, veri 1.1 TB; Dropbox sürümü kodla indirilemiyordu |
| 3 Türkiye test seti | İki yangın için önce/sonra 9 bantlık 20 m yığınlar, SCL maskesi, geçerli piksel maskesi, EFFIS referansı; FLOGA ile bant uyumu kontrolü | Bütün yöntemler aynı ızgara, maske ve referansla ölçülsün |
| 4 Benchmark'lar | dNBR (USGS eşikleri + FLOGA doğrulamasında seçilen eşik); BAM-CD v1 ve v2 resmî ağırlıklarla; önce FLOGA testinde makaledeki değere yakın mı diye kontrol edildi | Adil karşılaştırma (ders bölüm 5.3) |
| 5 Modelimiz | 36 özellik; Optuna ile HPO (sadece FLOGA doğrulama); 5 tohumla son eğitim; eşik her tohum için doğrulamada seçildi | Düşük maliyetli, GPU'suz yöntem |
| 6 Değerlendirme | Tüm yöntemler × (FLOGA test, Manavgat, Marmaris, birleşik Türkiye); blok bootstrap güven aralıkları; maliyet tablosu; ablasyon; arazi örtüsüne göre hata analizi | Ders bölüm 5.5: belirsizlik, ablasyon, ölçülmüş avantaj |
| 7 Kontrol etiketleme | 300 noktalık kör etiketleme kiti (`labeling/`) | EFFIS'ten bağımsız ikinci bir kontrol (isteğe bağlı insan işi) |
| 8 Demo | Colab'de tek tıkla çalışan notebook (≤ 30 dk) | Ders bölüm 6.3 |
| 9–12 | README, veri manifestosu, sürüm v1.0, IEEE makalesi, sunum taslağı, bu rapor | Teslimler |

## 4. Sonuçlar

### 4.1 Ana tablo (otomatik)
<!-- RESULTS:START -->
<!-- RESULTS:END -->

### 4.2 Yorum (otomatik sayılarla)
<!-- INTERPRETATION:START -->
<!-- INTERPRETATION:END -->

## 5. Sınırlılıklar (dürüstçe)
- **Referans bağımsızlığı:** EFFIS sınırları da Sentinel-2'den çıkarılmıştır. Yöntem farklı olsa da aynı
  sensörü kullanır; bu yüzden sonuçlar EFFIS'e "benzerliği" ölçer. Kontrol etiketleme kiti bu açığı kapatmak için
  hazırlandı ama henüz doldurulmadı.
- **Sadece iki yangın, tek mevsim:** Sonuçlar 2021 yazının iki büyük yangınına aittir; küçük yangınlar ve başka
  mevsimler denenmedi.
- **FLOGA sürümü:** Biz FLOGA'nın yeniden işlenmiş v2 görüntülerini kullandık; BAM-CD v1 ise orijinal v1
  görüntülerle eğitildi. Ofseti çıkararak uyumu sağladık, ama küçük farklar olabilir. FLOGA'nın bazı olaylarının v2
  görüntüsü yok (338 olaydan 327'si kullanıldı) ve yama seçimimiz makaledekinin birebir aynısı değil (aynı kurallar,
  farklı rastgele seçim).
- **Değerlendirme alanı:** EFFIS poligonunun sınır kutusu + 3 km tampon. Daha geniş alanlarda yanlış alarm oranı
  değişebilir.
- **Piksel bazlı model:** LightGBM komşu piksellere bakmaz; bu yüzden "tuz-biber" gürültüsü üretebilir.

## 6. Nasıl yeniden çalıştırılır?
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_all.py          # bitmiş adımları atlar; --force hepsini yeniden yapar; --list durumu gösterir
pytest -q
```
Colab demosu: README'deki "Open in Colab" düğmesi → *Runtime → Run all*.

## 7. Dosya listesi (en önemlileri)
| Dosya | Ne işe yarar |
|---|---|
| `configs/default.yaml` | Yangın kimlikleri, tarih aralıkları, bantlar, maske sınıfları |
| `configs/best_params_*.yaml` | HPO sonucu en iyi ayarlar |
| `src/burnt/stac.py` | Planetary Computer'dan sahne arama, geçiş gruplama, mozaik |
| `src/burnt/preprocess.py` | Ofset, yansıtım, SCL maskesi, 20 m'ye yeniden örnekleme |
| `src/burnt/indices.py`, `features.py` | Spektral indeksler, 36 özellik ve gruplar |
| `src/burnt/reference.py` | EFFIS poligonlarını okuma, rasterleştirme |
| `src/burnt/floga.py`, `remote_tiff.py` | FLOGA'yı uzaktan okuma, resmî bölme, yamalar |
| `src/burnt/models.py` | LightGBM / RF / LR eğitme, kaydetme, eşik seçimi |
| `src/burnt/bamcd.py` | BAM-CD'nin resmî kodu etrafında sarmalayıcı (kayan pencere) |
| `src/burnt/metrics.py` | Metrikler ve blok bootstrap |
| `src/burnt/cost.py` | Süre ve bellek ölçümü |
| `scripts/05…95_*.py` | Boru hattının numaralı adımları (README'de sırası var) |
| `scripts/demo_run.py`, `notebooks/00_demo_end_to_end.ipynb` | Demo |
| `tests/` | Birim testleri, sızıntı (leakage) testi |
| `outputs/tables/`, `outputs/figures/` | Bütün sonuç tabloları ve şekiller |
| `paper/main.pdf` | IEEE makalesi |
| `reports/` | Sunum taslağı, bu rapor, soru-cevap rehberi, yapılacaklar |
