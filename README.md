# VAE ile MNIST'te Görüntü Üretimi ve Gizli Uzay Analizi

Üretken Yapay Zeka dersi proje ödevim. MNIST el yazısı rakamları üzerinde, 2 boyutlu gizli uzaya sahip bir Variational Autoencoder (VAE) eğittim. Gizli uzayı çizdim, bu uzaydan yeni rakamlar ürettim, orijinal görüntülerle yeniden oluşturulanları karşılaştırdım ve iki nokta arasında interpolasyon yaptım. Kod PyTorch ile yazıldı.

## Veri seti

MNIST: 28x28 boyutunda gri tonlamalı el yazısı rakamlar. Notebook'ta 60.000 eğitim görüntüsünü kullandım. Veri `torchvision` ile notebook çalışırken otomatik iniyor, repoda veri yok.

Piksel değerlerini `ToTensor()` ile 0-1 aralığına çektim ve her görüntüyü 784 uzunluğunda vektöre düzleştirdim. Batch boyutu 128.

## Model

| Bölüm | Katman |
|-------|--------|
| Encoder | Linear(784, 400) + ReLU |
| Ortalama (mu) | Linear(400, 2) |
| Log-varyans (logvar) | Linear(400, 2) |
| Decoder | Linear(2, 400) + ReLU, Linear(400, 784) + Sigmoid |

Örnekleme için yeniden parametrizasyon hilesini kullandım: `z = mu + eps * std`. Kayıp fonksiyonu, toplam BCE ile KL ıraksamasının toplamı. Optimizer Adam, öğrenme oranı 0.001, eğitim 10 epoch.

## Eğitim

Epoch başına örnek başına ortalama kayıp (BCE + KL):

| Epoch | Kayıp | Epoch | Kayıp |
|-------|-------|-------|-------|
| 1 | 189.77 | 6 | 157.87 |
| 2 | 167.16 | 7 | 156.78 |
| 3 | 163.68 | 8 | 155.77 |
| 4 | 161.20 | 9 | 154.99 |
| 5 | 159.31 | 10 | 154.27 |

![Kayıp eğrisi](results/loss_curve.png)

## Sonuçlar

**Gizli uzay.** 60.000 eğitim görüntüsünün `mu` değerlerini rakam etiketine göre boyadım. 1, 6, 2 ve 7 rakamları az çok ayrı bölgelerde toplanıyor. 0, 3, 4, 5, 8 ve 9 ise ortada büyük ölçüde iç içe geçiyor.

![Gizli uzay](results/latent_space.png)

**Manifold.** Gizli uzayda -3 ile 3 arasında 15x15'lik bir ızgara oluşturup her noktayı decoder'a verdim. Komşu noktalar birbirine benzeyen rakamlar üretiyor, rakamlar arasında ani sıçrama yerine yavaş bir değişim var.

![Manifold](results/latent_manifold.png)

**Yeniden oluşturma.** Üst satır orijinal, alt satır modelin çıktısı. 784 boyutlu görüntü 2 sayıya sıkıştığı için çıktılar bulanık. Çoğu rakamın genel şekli korunuyor, ama bazıları başka rakama dönüşüyor (4 → 9, 5 → 8 gibi). Gösterilen görüntüler eğitim setinin ilk batch'inden.

![Yeniden oluşturma](results/reconstruction.png)

**İnterpolasyon.** Gizli uzaydan standart normal dağılımla rastgele iki nokta seçip aralarında 10 adım attım. Geçiş yumuşak ama görüntüler bulanık. Noktalar rastgele seçildiği için her çalıştırmada sonuç değişiyor.

![İnterpolasyon](results/interpolation.png)

## Eksikler

- Notebook'ta yalnızca tek bir ayarla (öğrenme oranı 0.001, batch 128, gizli boyut 2) 10 epoch eğitim var. Farklı hiperparametre denemeleri bu notebook'ta yok.
- Test seti yüklenmiyor, sonuçların hepsi eğitim verisi üzerinde. Doğrulama kaybı da izlenmedi.
- BCE ve KL terimlerini ayrı ayrı kaydetmedim, sadece toplam kaybı yazdırdım.
- Kayıp grafiğini epoch kayıplarını elle yazdığım bir listeden çizdim (eğitim döngüsünde listeye kaydetmemiştim). Değerler eğitim çıktısıyla aynı.
- Model sadece tam bağlı (Linear) katmanlardan oluşuyor, CNN kullanmadım.

## Klasörler

- `notebooks/vae_mnist.ipynb`: kodun kendisi
- `results/`: notebook çıktılarından aldığım grafikler

## Çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook notebooks/vae_mnist.ipynb
```

MNIST ilk çalıştırmada kendiliğinden iniyor. GPU varsa kullanıyor, yoksa CPU'da da çalışır.

Taha Doğukan Bahadır
