# Size kalan işler (insan gerektiren)

Otomatik kısım bitti. Aşağıdakiler yalnızca ekip üyelerinin yapabileceği işler. Önem sırasına göre:

1. **Görev dağılımını (CRediT) yazın — ZORUNLU.** Makalenin "Author Contributions" bölümünde
   `[TODO: confirm roles]` yazıyor. Ara sunumda kimin neyi yaptığını söylediyseniz onu yazın
   (`paper/sec_body.tex`). Ders kuralı: "herkes kod yazar, herkes yazar". Sonra `tectonic paper/main.tex` ile
   PDF'i yeniden üretin. Sunum taslağındaki konuşmacı paylaşımını da buna göre düzeltin.
2. **Erdem'in kendi commit'leri.** Notlandırmada depo geçmişine bakılıyor. Erdem kendi hesabıyla en az bir anlamlı
   katkı (ör. makale metninde düzeltme, sunum slaytları, kontrol etiketleri) commit'lemeli.
3. **Makaleyi kendi cümlelerinizle okuyup düzeltin.** Metin Claude Code ile taslak olarak yazıldı; yapay zekâ
   kullanım beyanı makalede var. Her cümleyi anladığınızdan ve savunabildiğinizden emin olun (sözlü kontrol!).
4. **Ara sunum geri bildirimi.** Hocanın 23 Kasım yorumları varsa listeleyin ve makalede karşılandığını kontrol
   edin (dosya bulunamadığı için otomatik ele alınamadı).
5. **(İsteğe bağlı) Kontrol etiketlemesi.** `labeling/NASIL_ETIKETLENIR.md` adımlarını izleyerek 300 noktayı
   etiketleyin (~2–3 saat, iki kişi). Sonra `python scripts/75_eval_checklabels.py` → `python run_all.py`.
6. **Sunum provası.** `reports/final_presentation_outline.md`: 12 dk + 3 dk demo. Demo için yedek video kaydedin;
   Colab'i sunumdan ~1 saat önce bir kez çalıştırın. `reports/SUNUM_VE_SORU_CEVAP_TR.md` içindeki soruları
   birbirinize sorun.
7. **Teslim (4 Ocak 2027 23:59):** `paper/main.pdf` + depo bağlantısı + release `v1.0`. Son değişikliklerden sonra
   gerekiyorsa `v1.0.1` etiketi atın (`gh release create v1.0.1 ...`).
8. **Colab'de demo süresini bir kez ölçün.** README'deki "Open in Colab" ile not defterini CPU modunda
   çalıştırın ve yazdırılan toplam süreyi `docs/demo_runtime.md` dosyasına ekleyin (yerelde ölçülen 186 s,
   Colab 2 çekirdek tahmini < 15 dk).
9. **Akran değerlendirmesi** formlarını doldurun (gizli).
