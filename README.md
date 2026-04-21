
---

# 🌲 Forest Cover Type Classification (README.md)

```markdown
# 🌲 Forest Cover Type Classification

Bu projede, çevresel özellikler kullanılarak orman örtüsü türlerinin makine öğrenmesi ile tahmin edilmesi amaçlanmıştır.

---

## 📌 Proje Hakkında

Forest Cover Type veri seti kullanılarak farklı makine öğrenmesi modelleri uygulanmış ve karşılaştırılmıştır.

Amaç: En iyi performansı veren modeli belirlemek.

---

## 📊 Veri Seti

- Forest Cover Type Dataset
- Her gözlem: 30x30 m arazi
- Hedef: Cover_Type

### Özellikler

- Elevation
- Slope
- Aspect
- Su kaynaklarına uzaklık
- Yol uzaklığı
- Diğer çevresel değişkenler

---

## ⚙️ Veri Ön İşleme

- Eksik değer analizi yapıldı
- Target eksik olan satırlar silindi
- Median ile eksik değer doldurma
- %80 / %20 train-test split
- StandardScaler uygulandı
- 10-Fold Cross Validation kullanıldı

---

## 🤖 Kullanılan Modeller

- Logistic Regression
- Random Forest
- KNN
- Hybrid Model (LogReg + Random Forest)

---

## 📈 Değerlendirme Metrikleri

- Accuracy
- Precision
- Recall
- F1 Score

---

## 🔁 Cross Validation

- 10-Fold Stratified CV uygulandı
- Ortalama performans hesaplandı

---

## 📊 Görselleştirmeler

- Hedef değişken dağılımı
- Confusion Matrix
- Model karşılaştırma grafiği
- Feature Importance
- PCA grafiği

---

## 🏆 Sonuçlar

- Random Forest en başarılı modeldir
- Hybrid model stabil sonuçlar üretmiştir
- Logistic Regression daha düşük performans göstermiştir
- KNN orta seviyede performans göstermiştir

---

## 📉 PCA Analizi

- Veri 2 boyuta indirgenmiştir
- Görselleştirme yapılmıştır
- Sınıflar arasında örtüşmeler gözlemlenmiştir

---

## 🧪 Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 Kurulum ve Çalıştırma

```bash
git clone https://github.com/kullaniciadi/forest-project.git
cd forest-project
pip install -r requirements.txt
python main.py
