# Makine Öğrenmesi Tahmin Modelleri

![giphy](https://media.giphy.com/media/3ornk57KwDXf81rjWM/giphy.gif)

## Giriş:
Makine öğrenmesi, günümüzde verilerin analizi ve geleceği tahmin etme süreçlerinde vazgeçilmez bir rol oynamaktadır. Veri analizi ve tahmin, hem iş dünyasında hem de bilimsel araştırmalarda kullanılan önemli araçlardır. Bu makalede, veri analizi ve tahmin süreçlerinde sıkça kullanılan makine öğrenimi tahmin modellerini inceleyeceğiz.

Verilerin doğası gereği, iki ana kategori altında incelenebilir: Kategorik ve Sayısal. Kategorik veriler, genellikle sınıflandırmada kullanılır ve iki alt kategoriye ayrılır: Nominal ve Ordinal. Sayısal veriler ise tahminlerde kullanılır ve iki alt kategoriye ayrılır: Oransal(Ratio) ve Aralık(Interval)

- Nominal Veriler: Bu kategori, sıralaması ve önceliği olmayan verileri ifade eder. Örneğin, renkler veya ürün kategorileri nominal verilere örnek olarak verilebilir.
- Ordinal Veriler: Bu kategori, verilerin belirli bir sıralamaya sahip olduğu durumları ifade eder. Örneğin, eğitim seviyeleri(ilkokul, lise, üniversite) ordinal verilere örnek olarak verilebilir.

- Oransal(Ratio) Veriler: Oransal veriler, sıfır noktasına sahip verileri ifade eder. Örneğin bir kişinin geliri oransal bir veri türüdür. Çünkü sıfır gelir, gerçek bir anlam taşır.

- Aralık(Interval) Veriler: Aralık verileri, sıfırın gerçek bir anlam taşımadığı verileri ifade eder. Örneğin, sıcaklık aralığı aralık bir veri türüdür, çünkü sıcaklığın sıfır derece olması soğukluğun tamamen yok olduğu anlamına gelmez.

Bu makalede , özellikle sayısal veriler üzerindeki tahmin modellerine odaklanacağız. Tahmin, hem geçmiş verilere dayalı olarak geleceği tahmin etme(Prediction) sürecini içerebilir hem de ileriye yönelik verileri tahmin etme (Forecasting) amacını taşıyabilir.

Bu makalede, özellikle **Lineer Regresyon, Polinom Regresyon, Destek Vektör Regresyon, Karar Ağacı Regresyon ve Rassal Ağaçlar Regresyon** gibi çeşitli tahmin modellerini inceleyeceğiz.

## VERİ SETİ VE ÖN İŞLEME
### Veri Seti Tanımı
Burada, çeşitli tahmin modellerini incelemek ve uygulamak için farklı veri setleri kullanılmıştır.

### 1. Satışlar Veri Seti('satislar.csv')
Bu veri seti, aylık satış verilerini içerir ve iki ana sütun bulunur.
- Aylar: Satış verilerinin aylara göre sıralandığı sütundur.
- Satışlar: Aylık satış miktarını gösteren sayısal verileri içerir.

### 2. Ülke Veri Seti('veriler.csv')
Bu veri seti, bir kişinin ülke, boy, kilo, yaş ve cinsiyet bilgilerini içerir.Her bir sütun şu şekildedir:

- **Ülke:** Kişinin yaşadığı ülkeyi temsil eder (Nominal veri türü).
- **Boy:** Kişinin boyunu ölçen sayısal bir değerdir (Oransal veri türü).
- **Kilo:** Kişinin kilosunu ölçen sayısal bir değerdir (Oransal veri türü).
- **Yaş:** Kişinin yaşıdır (Oransal veri türü).
- **Cinsiyet:** Kişinin cinsiyetini temsil eder (Nominal veri türü).

### 3.Hava Durumu Veri Seti('tenis.csv')
Bu veri seti, hava koşulları ile tenis oynama kararları arasındaki ilişkiyi incelemek içindir.Veri seti şu sütunlardan oluşur:
- **Hava Durumu:** Hava koşullarını temsil eden bir sütundur.(Nominal Veri TÜrü)
- **Sıcaklık:** Sıcaklığı temsil eden bir sayısal değerdir(Oransal Veri Türü)
- **Nem:** Nem seviyesini temsil eden sayısal bir değerdir(Oransal Veri TÜrü)
- **Rüzgarlı** Rüzgarlı hava durumunu temsil eden bir sütundur(Nominal Veri Türü)
- **Tenis Oynama Durumu** Tenis oynama kararını gösteren sütundur(Nominal Veri Türü)

Bu veri seti, hava koşullarına göre tenis oynama olasılığını tahmin etmek için kullanılacaktır.

## 4.Maaşlar Veri Seti('maaslar.csv')
Bu veri seti, çalışanların unvan,eğitim seviyesi ve maaş bilgilerini içerir. Sütunlar:

- **Unvan:** Çalışanın Unvanını Temsil Eden Sütundur(Nominal Veri Türü)

- **Eğitim Seviyesi:** Çalışanın eğitim seviyesini temsil eden sütundur(Oransal Veri Türü)
- **Maaş:** Çalışanın maaşını gösteren sayısal bir değerdir.(Oransal veri türü)

Bu veri seti, çalışanların maaşlarını eğitim seviyelerine göre tahmin etmek için kullanılacaktır.

## 5.Yeni Maaşlar Veri Seti('yenimaaslar.csv')
Bu veri seti, çeşitli faktörlere (çalışan ID'si, Unvan, Unvan Seviyesi, Kıdem ve Puan) dayalı olarak çalışanların maaşlarını içerir. Veri Seti şunları içerir:

- **Çalışan ID:** Her çalışanı benzersiz bir şekilde tanımlayan sütundur.
- **Unvan:** Çalışanın Unvanını temsil eden sütundur(Nominal Veri Türü)
- **Unvan Seviyesi:** Çalışanın unvan seviyesini temsil eden sütundur.(Oransal Veri Türü)
- **Kıdem:** Çalışanın Kıdemini temsil eden sayısal bir değerdir(Oransal veri türü)
- **Puan:** Çalışanın performansını temsil eden sayısal bir değerdir(Oransal Veri Türü)
- **Maaş:** Çalışanın maaşını gösteren sayısal bir değerdir.(Oransal Veri Türü)

Bu veri seti, çalışanların maaşlarını unvan seviyelerine, kıdemlerine ve performanslarına göre tahmin etmek için kullanılacaktır.

## Lineer Regresyon
### Adım 1: Gerekli Kütüphaneleri İçe Aktarma

İlk adım, bu uygulamada kullanacağımız Python kütüphanelerini içe aktarmaktır. Bu kütüphaneler, veri işleme, model oluşturma ve sonuçları görselleştirme için gereklidir.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
```
Bu kütüphaneler, verileri işlemek, bir tahmin modeli oluşturmak ve sonuçları görselleştirmek için kullanılacak.

### Adım 2: Verileri Okuma
Şimdi, projemiz için kullanacağımız verileri içe aktaralım. Bu örnekte, "satislar.csv" adlı bir veri dosyasını kullanacağız ve bu verileri bir Pandas DataFrame'e aktaracağız.
```python
veriler = pd.read_csv("datasets\satislar.csv")
```
![veriler](https://i.imgur.com/tg5Lnes.jpeg)

Verileri içe aktardıktan sonra, bunları daha fazla işlemek ve analiz etmek için kullanabiliriz.

### Adım 3: Verileri Hazırlayın

Verileri tahmin modeline hazırlamadan önce, hangi sütunların kullanılacağını ve nasıl ayrılacağını belirlememiz gerekiyor. Bu projede, "Aylar" ve "Satislar" sütunlarını kullanarak satış tahmini yapacağız.
```python
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]
```
![aylar](https://i.imgur.com/j11rYy4.jpeg) ![satışlar](https://i.imgur.com/DGJZfhH.jpeg)

Daha sonra, verileri eğitim ve test veri kümelerine ayırarak modelimizi eğitmek ve test etmek için hazır hale getiriyoruz.
```python
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)
```
### Adım 4: Verileri Ölçeklendirme

Verilerin ölçeklendirilmesi, modelin daha iyi performans göstermesine yardımcı olabilir. Bu nedenle, StandardScaler kullanarak verileri ölçeklendiriyoruz.

```python
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
```
![train__test](https://i.imgur.com/xAOYttG.jpeg) 
### Adım 5: Modeli İnşa Etme

Şimdi, verileri hazırladık ve model oluşturma zamanı geldi. Bu projede, basit bir lineer regresyon modeli kullanıyoruz.
```python
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)
x_train = pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)

x_train=x_train.sort_index()
y_train=y_train.sort_index()
```
Modeli eğittikten sonra, veriler üzerinde tahminler yapabiliriz.

### Adım 6: Tahmin ve Görselleştirme

Modelimizi kullanarak tahminlerde bulunuyoruz ve bu tahminleri görselleştiriyoruz. Bu, modelin ne kadar iyi çalıştığını anlamamıza yardımcı olur.
```python
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,lr.predict(x_test),color='blue')
plt.title("Aylara Göre Satış")
plt.show()
```
![aylaragöresatışgrafik](https://i.imgur.com/qiKXpRW.jpeg) 
Bu grafik, aylara göre satışların nasıl tahmin edildiğini göstermektedir.

## Multiple Regression(Çoklu Regresyon)
Çoklu regresyon, birden fazla bağımsız değişkenin bağımlı değişken üzerindeki etkisini incelemek için kullanılan güçlü bir istatistiksel yöntemdir.

### Çoklu Regresyon Modeli Denklemi
Çoklu regresyon modeli, aşağıdaki denkleme dayanır:

ŷ = β0 + β1⋅x1 + β2⋅x2 + β3⋅x3 + ɛ
Bu denklemin açıklamaları:
- ŷ, bağımlı değişkeni temsil eder.
- x1, x2, ve x3, bağımsız değişkenleri temsil eder.
- β0, β1, β2, β3, regresyon katsayılarını temsil eder.
- ɛ, hata terimini temsil eder.

### Adım 1:Veri Ön İşleme

Her veri analizi projesinde olduğu gibi, başlangıçta veri ön işleme adımı gereklidir. İşte bu projede kullanacağınız veri ön işleme adımları:

#### Veriyi İçe Aktarma
```python
veriler = pd.read_csv("datasets\veriler.csv")
```
- Kategorik Değişkenleri Sayısal Değişkenlere Dönüştürme
```python
from sklearn import preprocessing
Yas=veriler.iloc[:,1:4].values
ulke = veriler.iloc[:, 0:1].values
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])
```
Ardından, One-Hot Encoding kullanarak bu sayısal değerleri çoklu sütunlara dönüştürürüz:
```python
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
```

```python
c = veriler.iloc[:,-1:].values
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
c=ohe.fit_transform(c).toarray()
print(c)
```
![c](https://i.imgur.com/BZvB3zV.jpeg)  
- DataFrame Birleştirme
```python
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
cinsiyet = veriler.iloc[:, -1].values
sonuc3 = pd.DataFrame(data=c[:,:1], index=range(22), columns=['cinsiyet'])
s1 = pd.concat([sonuc, sonuc2], axis=1)
s2 = pd.concat([s1, sonuc3], axis=1)
```
![Dataframe](https://i.imgur.com/qTOoZDj.jpeg)
### Adım 2: Veri Analizi ve Çoklu Regresyon</h3>
Şimdi verileri eğitim ve test veri kümelerine böleriz ve çoklu regresyon analizini uygularız:
```python
x_train, x_test, y_train, y_test = train_test_split(s1, sonuc3, test_size=0.33, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
```
Benzer bir şekilde, bağımlı değişkeni olan boy için bir regresyon analizi yapabiliriz:
```python
boy = s2.iloc[:, 3:4].values
sol = s2.iloc[:, :3]
sag = s2.iloc[:, 4:]
veri = pd.concat([sol, sag], axis=1)
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)
regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)
y_pred = regressor2.predict(x_test)
```
### Geriye Eleme Yöntemi
```python
import statsmodels.api as sm

X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

x_l=veri.iloc[:,[0,1,2,3]].values
x_l=np.array(x_l,dtype=float)

model=sm.OLS(boy,x_l).fit()
print(model.summary())
```
![OLS_REGRESSİON](https://i.imgur.com/43DoUPc.jpeg)
Şimdi bununla ilgili bir örnek yapalım.
### Adım 1: İlk olarak gerekli kütüphaneleri içe aktarıyoruz.
```python
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import r2_score
```
### Adım 2: Veri yükleme
```python
veriler = pd.read_csv("datasets\odev_tenis.csv")
```
![tenis](https://i.imgur.com/MVEAyCU.jpeg)
### Adım 3: Veri ön işleme

#### Kolay yöntemle kategorik verileri numerik hale getirme
```python
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
```
# One-Hot Encoding işlemi
```python
c = veriler2.iloc[:, :1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
havadurumu = pd.DataFrame(data=c, index=range(14), columns=['o', 'r', 's'])
sonveriler = pd.concat([havadurumu, veriler.iloc[:, 1:3]], axis=1)
sonveriler = pd.concat([veriler2.iloc[:, -2:], sonveriler], axis=1)
```
![sonveriler](https://i.imgur.com/vcjjKWB.jpeg)
### Adım 4: Veriyi eğitim ve test verilerine bölmek
```python
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:, :-1], sonveriler.iloc[:, -1:], test_size=0.33, random_state=0)
```
### Adım 5: Lineer Regresyon modeli oluşturma ve eğitme
```python
regressor = LinearRegression()
regressor.fit(x_train, y_train)
```
### Adım 6: Tahmin yapma
```python
y_pred = regressor.predict(x_test)
```
### Adım 7: İstatistiksel analiz yapma
```python
X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)
x_l = sonveriler.iloc[:, [0, 1, 2, 3, 4, 5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:, -1:], x_l).fit()
```
### Adım 8: İstatistiksel analiz sonuçlarını görüntüleme
```python
print(model.summary())
```
![OLS](https://i.imgur.com/BpriYhU.jpeg)
### Adım 9: Gereksiz sütunları kaldırma ve yeniden analiz yapma
```python
sonveriler = sonveriler.iloc[:, 1:]
X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)
x_l = sonveriler.iloc[:, [0, 1, 2, 3, 4]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:, -1:], x_l).fit()
```
### Adım 10: İkinci analiz sonuçlarını görüntüleme
```python
print(model.summary())
```
![OLS](https://i.imgur.com/is6UAhw.jpeg)
### Adım 11: Gereksiz sütunları kaldırıp modeli tekrar eğitme ve tahmin yapma
```python
x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
```
## Polynomial Regression
### Adım 1: Gerekli kütüphaneleri içe aktarma
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
```

### Adım 2: Veriyi yükleme
```python
veriler = pd.read_csv("datasets\maaslar.csv")
```
![maaslar.csv](https://i.imgur.com/lxS6IYp.jpeg)
### Adım 3: Veriyi dilimleme
```python
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]

```
![değişkenler](https://i.imgur.com/MtE4XgG.jpeg)
### Adım 4: Doğrusal Regresyon (Linear Regression)

#### Doğrusal model oluşturma

```python
lineer_regression = LinearRegression()
lineer_regression.fit(x.values, y.values)
```

#### Görselleştirme
```python
plt.scatter(x.values, y.values, color='red')
plt.plot(x, lineer_regression.predict(x.values), color='blue')
plt.title('Doğrusal Regresyon')
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Maaş')
plt.show()
```
![görselleştirme](https://i.imgur.com/2ML3RnI.jpeg)
### Adım 5: Polinomal Regresyon (Polynomial Regression)
#### Doğrusal olmayan model oluşturma
```python
poly_regression = PolynomialFeatures(degree=2)
x_poly = poly_regression.fit_transform(x.values)

lineer_regression2 = LinearRegression()
lineer_regression2.fit(x_poly, y)
```
#### Görselleştirme
```python
plt.scatter(x.values, y.values, color='red')
plt.plot(x.values, lineer_regression2.predict(poly_regression.fit_transform(x.values)), color='blue')
plt.title('Polinomal Regresyon (Derece 2)')
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Maaş')
plt.show()
```
![görselleştirmepolinom](https://i.imgur.com/n79ERMr.jpeg)
## Support Vektor Regression
### Giriş
Bu bölümde, veri analizi ve makine öğrenimi kullanarak bir maaş tahmin modeli geliştirmede Destek Vektör Modeli adımlarını anlatacağım.

### Adım 1: Verilerin Yüklenmesi
ilk olarak, maaş verilerini bir csv dosyasından yükleyelim.
```python
veriler=pd.read_csv("datasets\maaslar.csv")
```
### Adım 2: Veri Hazırlığı

Veri analizi için verileri hazırlamamız gerekir. Bu adımda,  veri dilimliyoruz ve ölçeklendiriyoruz. x ve y değişkenlerini ayrıştırıyoruz ve ölçeklenmiş verileri kullanıyoruz.
```python
#dataframe dilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli=sc1.fit_transform(x.values)

sc2=StandardScaler()
y_olcekli=sc2.fit_transform(y.values)

```
![xveytrain](https://i.imgur.com/7uFRrf3.jpeg)
### Adım 3: Support Vektor Regresyon
Şimdi, SVR algoritmasını kullanarak maaş tahmin modelimizi oluşturuyoruz. SVR'nin çekirdek(kernel) fonksiyonu 'rbf' olarak ayarlanmıştır.
```python
##Support Vektor Regression import
from sklearn.svm import SVR 
svr_reg=SVR(kernel='rbf')

svr_reg.fit(x_olcekli,y_olcekli)
```
### Adım 4: Sonuçların Görselleştirilmesi
Modelimizi eğittikten sonra, sonuçları görselleştirmek önemlidir.Aşağıda, ölçeklenmiş verilerin üzerine modelin tahminlerini yerleştirdiğimiz bir grafik bulunmaktadır.
```python
import matplotlib.pyplot as plt

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
```
![svrgorselleştirme](https://i.imgur.com/2JkND7h.jpeg)
## Decision Tree (Karar Ağacı) Regression
Bu bölümde, Karar Ağacı Regresyonu kullanarak maaş tahmin modelimizi oluşturalım. Karar ağacı, verileri karar düğümleri ve yaprak düğümleri aracılığıyla sınıflandırmak ve regresyon yapmak için kullanılan güçlü bir algoritmadır.

### Adım 1: Veri hazırlığı ve Eğitme

```python
# Karar Ağacı Regresyonu import
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x.values, y.values)
```
### Adım 2: Sonuçların Görselleştirilmesi
Karar ağacı modelini eğittikten sonra, sonuçları görselleştirmek için aşağıdaki grafik kullanılabilir.Karar ağacı tahminlerini veri noktalarına uygulayarak modelin nasıl performans gösterdiğini görebiliriz.
```python
import matplotlib.pyplot as plt

plt.scatter(x.values, y.values, color='red')
plt.plot(x.values, dt_reg.predict(x.values), color='blue')
plt.title("Decision Tree")
plt.xlabel("Deneyim(Yıl)")
plt.ylabel("Maaş")
plt.show()
```
![decisiontree](https://i.imgur.com/zXhBvqa.jpeg)
## Random Forest(Rassal Ağaçlar)
Veri analitiği için sıkça kullanılan bir makine öğrenme algoritması olan Rassal Ağaçlar kullanılarak bir regresyon modeli geliştirmeyi öğrenelim.

### Adım 1: Verileri Yükleme
Gerekli olan verileri csv dosyasından yükleyelim.
```python
import pandas as pd

veriler = pd.read_csv("datasets\maaslar.csv")
```
### Adım 2: Veri Dilimleme
Verilerimizi bağımsız değişken (x) ve bağımlı değişken (y) olarak  ayıralım.
```python
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
```
### Adım 3: Verilerin Ölçeklenmesi
Verilerimizi ölçeklendirerek, makine öğrenme algoritmasının daha iyi performans göstermesini sağlayalım.
```python
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x.values)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y.values)
```
### Adım 4: Rassal Ağaçlar Regresyon Modeli Oluşturma
Rassal ağaçlar regresyon modelini oluşturalım ve eğitelim

```python
from sklearn.ensemble import RandomForestRegressor

rf_regression = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regression.fit(x.values, y.values)
```
### Adım 5: Sonuçları Görselleştirme
Oluşturduğumuz regresyon modelinin sonuçlarını görselleştirelim.

```python
import matplotlib.pyplot as plt

plt.scatter(x.values, y.values, color='red')
plt.plot(x.values, rf_regression.predict(x.values), color='blue')
plt.xlabel("Deneyim(Yıl)")
plt.ylabel("Maaş")
plt.title("Random Forest ")
plt.show()
```
![randomforest](https://i.imgur.com/hfB9IyY.jpeg)

![giphy2](https://media.giphy.com/media/yXOiYLbYecDIaXjlqq/giphy.gif)

**Bir Uygulama Daha Yapalım.**

- **Veri Kümesini İndirelim, Gerekli/Gereksiz Bağımsız Değişkenleri Bulalım.**
- **5 Farklı Yönteme Göre Regresyon Modellerini Çıkaralım.**
- **Yöntemlerin başarılarını karşılaştıralım.**
## UYGULAMA 2 

### Adım 1: Veri Yükleme
İlk olarak uygulamamız için gerekli verileri csv dosyasından yükleyelim.
```python
import pandas as pd

veriler = pd.read_csv("datasets\yenimaaslar.csv")
```
![yenimaaslar](https://i.imgur.com/pZcvi9P.jpeg)
### Adım 2: Veri Dilimleme
Verilerimizi bağımsız değişken(x) ve bağımlı değişken(y) olarak  ayıralım.
```python
x = veriler.iloc[:, 2:3]
y = veriler.iloc[:, 5:]
```
![xveydegerleri](https://i.imgur.com/5PLPvQf.jpeg)
### Adım 3: Veri Keşfi
Verilerin arasındaki ilişkileri incelemek için basitçe korelasyon analizi yapalım.
```python
print(veriler.corr())
```
![corrveriler](https://i.imgur.com/ZrKe3Qo.jpeg)
### Adım 4: Regresyon Modelleri
Verileri kullanarak farklı  regresyon modelleri oluşturalım.

#### Adım 4.1 Linear Regression
```python
lineer_regression=LinearRegression()
lineer_regression.fit(x.values,y.values)

model1=sm.OLS(lineer_regression.predict(x.values),x.values)

print(model1.fit().summary())

```
![lineerregressionols](https://i.imgur.com/d835hUu.jpeg)
#### Adım 4.2 Polynomial Regression
```python
polynomial_reg=PolynomialFeatures(degree=2)
x_poly=polynomial_reg.fit_transform(x.values)

lineer_regression2=LinearRegression()
lineer_regression2.fit(x_poly,y.values)

plt.scatter(x.values, y.values,color='red')
plt.plot(x.values,lineer_regression2.predict(x_poly),color='blue')
plt.show()

model2=sm.OLS(lineer_regression2.predict(polynomial_reg.fit_transform(x.values)),x.values)
print(model2.fit().summary())

```
![polynomialregression](https://i.imgur.com/hMGlOir.jpeg)
#### Veri Ölçekleme
```python
scaler1=StandardScaler()
x_olcekli=scaler1.fit_transform(x.values)

scaler2=StandardScaler()
y_olcekli=scaler2.fit_transform(y.values)
```
#### Adım 4.3 Support Vektor Regression
```python
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()

model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
```
![svr](https://i.imgur.com/G6mOvWZ.jpeg)
#### Adım 4.4 Decision Tree
```python
from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color='red')
plt.plot(x.values,dt_reg.predict(x.values),color='blue')
plt.show()

model4=sm.OLS(dt_reg.predict(x.values),x.values)
print(model4.fit().summary())
```
![decisiontree](https://i.imgur.com/nc3p0fM.jpeg)
#### Adım 4.5 Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=3,random_state=0)
rf_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color='red')
plt.plot(x.values,rf_reg.predict(x.values),color='blue')

model5=sm.OLS(rf_reg.predict(x.values),x.values)
print(model5.fit().summary())
```
![randomforest](https://i.imgur.com/VsoTfSS.jpeg)
## Çözüm
- Gerekli: **Unvanseviye, Kıdem, Puan, Maas**
- Gereksiz: **Unvan, ID**

**Tek Parametreli Olarak:**
- Linear Regression R-Squared: **0.942**
- Polynomial Regression R-Squared: **0.810**
- Support Vector Regression R-Squared: **0.770**
- Decision Tree R-Squared: **0.751**
- Random Forest R-Squared: **0.721**

**Üç Parametreli Olarak:**
- Linear Regression R-Squared: **0.903**
- Polynomial Regression R-Squared: **0.680**
- Support Vector Regression R-Squared: **0.782**
- Decision Tree R-Squared: **0.679**
- Random Forest R-Squared: **0.713**

## Sonuç
Bu uygulamada farklı regresyon modellerini kullanarak farklı bağımsız değişkenlerle (Unvanseviye, Kıdem, Puan) bir işçinin maaşını tahmin etmeye çalıştık. İlk olarak, tüm bağımsız değişkenleri kullanarak tek parametreli regresyon modellerini oluşturduk ve bu modellerin başarılarını R-Squared değeri ile değerlendirdik.

***Tek Parametreli Modeller:***
- **Linear Regression  R-Squared (0.942):** Tek bir parametre kullanıldığında, model verilere çok iyi uyuyor. Ancak bu basit modelin gerçek dünyadaki karmaşıklığı yeterince yakalayamayabilir.

- **Polynomial Regression R-Squared (0.810):** Polinom regresyonun R-Squared değeri düşük, bu da modelin verilere uygun bir şekilde uymadığını gösterir.

- **Support Vector Regression(SVR) R-Squared (0.770)::**  Destek vektör regresyonu, verilere tek bir parametre kullanarak iyi bir uyum sağlar, ancak diğer modellere kıyasla daha düşük bir R-Squared değerine sahiptir.

- **Decision Tree Regression  R-Squared (0.751):**  Karar ağacı modeli tek bir parametre ile biraz daha düşük bir uyum sağlar, ancak hala kabul edilebilir bir değerdir.

- **Random Forest Regression R-Squared (0.721):** Random forest modeli tek bir parametre ile biraz daha düşük bir uyum sağlar, ancak yine de kabul edilebilir bir değerdir.

**Üç Parametreli Modeller:**

- **Linear Regression  R-Squared (0.903):** Üç parametre kullanıldığında, basit doğrusal regresyon modelinin R-Squared değeri yine yüksektir.

- **Polynomial Regression  R-Squared (0.680):** Polinom regresyonun R-Squared değeri üç parametre ile kullanıldığında artmaz, hatta düşer.

- **Support Vector Regression(SVR) R-Squared (0.782):** Destek vektör regresyonu, üç parametre kullanıldığında daha yüksek bir uyum sağlar.

- **Decision Tree Regression R-Squared (0.679):** Karar ağacı modelinin R-Squared değeri üç parametre ile kullanıldığında artmaz, hatta düşer.

- **Random Forest Regression R-Squared (0.713):** Random forest modeli, üç parametre ile kullanıldığında artmaz, hatta düşer.


![giphy3](https://media.giphy.com/media/doCy6bjRnZgTw6Oh2O/giphy.gif)
