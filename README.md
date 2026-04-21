🌲 Forest Cover Type Classification

Bu projede, çevresel ve coğrafi özellikler kullanılarak orman örtüsü türlerinin makine öğrenmesi algoritmaları ile tahmin edilmesi amaçlanmıştır.

📌 Proje Hakkında

Forest Cover Type veri seti, ABD’deki ormanlık alanlara ait çeşitli çevresel özellikleri içermektedir. Bu projede:

Farklı makine öğrenmesi algoritmaları uygulanmış
Modellerin performansları karşılaştırılmış
En iyi sonucu veren model belirlenmiştir

Amaç:

Çevresel veriler kullanılarak doğru ve genellenebilir bir sınıflandırma modeli geliştirmek

📊 Veri Seti
📁 Forest Cover Type Dataset
Her gözlem: 30x30 m arazi parçası
Hedef değişken: Cover_Type (orman türü)
İçerdiği özellikler:
Yükseklik (Elevation)
Eğim (Slope)
Bakı (Aspect)
Su kaynaklarına uzaklık
Yol uzaklığı
Toprak tipi ve diğer çevresel özellikler


⚙️ Veri Ön İşleme

Projede aşağıdaki adımlar uygulanmıştır:

Eksik değer analizi yapıldı
Target (Cover_Type) eksik olan satırlar silindi
Özelliklerdeki eksik veriler median ile dolduruldu
Train/Test split (%80 - %20, stratified)
StandardScaler ile ölçekleme (gereken modeller için)
10-Fold Cross Validation uygulandı


🤖 Kullanılan Modeller
Logistic Regression
Random Forest
K-Nearest Neighbors (KNN)
Hybrid Model (Logistic Regression + Random Forest - VotingClassifier)
📈 Değerlendirme Metrikleri
Accuracy
Precision (weighted)
Recall (weighted)
F1 Score (weighted)
🔁 Cross Validation
10-Fold Stratified Cross Validation kullanıldı
Her model için ortalama performans hesaplandı
Model karşılaştırması daha güvenilir hale getirildi


📊 Görselleştirmeler

Projede aşağıdaki görselleştirmeler yapılmıştır:

Hedef değişken dağılımı (class imbalance)
Confusion Matrix (her model için)
Model performans karşılaştırma grafiği
Feature Importance (Random Forest)
PCA (2 boyutlu görselleştirme)


🏆 Sonuçlar
Random Forest modeli en yüksek performansı göstermiştir
Hybrid model (LogReg + RF), stabil ve güçlü sonuçlar üretmiştir
Logistic Regression, karmaşık veri yapısında daha düşük performans göstermiştir
KNN modeli orta seviyede başarı sağlamıştır
📉 PCA Analizi
Yüksek boyutlu veri 2 boyuta indirgenmiştir
Veri dağılımı görselleştirilmiştir
Sınıflar arasında kısmi örtüşmeler gözlemlenmiştir

🧪 Kullanılan Teknolojiler
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn


🚀 Kurulum ve Çalıştırma
git clone https://github.com/kullaniciadi/proje-adi.git
cd proje-adi
pip install -r requirements.txt
python main.py

📁 Proje Yapısı
├── forest_cover_with_few_missing.csv
├── main.py
├── README.md
└── rapor.pdf


📌 Öne Çıkan Noktalar
Ensemble (Hybrid) model kullanımı
Cross Validation ile sağlam değerlendirme
Feature importance analizi
PCA ile veri görselleştirme


⚠️ Kısıtlar
Veri seti karmaşık ve yüksek boyutlu
Bazı sınıflar arasında benzerlik bulunmakta
Daha ileri modeller (MLP vb.) bu versiyonda kullanılmamıştır
