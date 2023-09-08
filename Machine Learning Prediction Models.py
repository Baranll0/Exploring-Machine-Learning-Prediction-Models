
##Multiple Regression
# y=a0+a1.x1+a2.x2+a3.x3+e

"""
p-value olasılık değeri
h0: farksızlık hipotezi, sıfır hipotezi
h1: alternatif hipotez
p-değeri: olasılık değeri(genelde 0.05 alınır)
p-değeri küçüldükçe h0 hatalı olma ihtimali artar
p-değeri büyüdükçe h1 hatalı olma ihtimali artar


----------------------------------------------

Çok Değişkende Değişken Seçimi:
    1.Yöntem: Bütün değişkenleri dahil etmek: 
    Şayet değişken seçimi yapıldıysa ve değişkenlerden zorunluluk varsa (örneğin, bankadaki kredi skorları için geliştirilen modelin başarısının ölçülmesi)
    keşif için(diğer 4 yöntemi kullanmadan önce bir ön fikir elde etmek için)

    2.Yöntem: Geriye doğru eleme(Backward Elimination):
        1.adım-Significance Level(SL) seçilir.(genelde 0.05)
        2.adım-Bütün değişkenler kullanılarak bir model inşa edilir.
        3.adım-En Yüksek p-value değerine sahip olan değişken ele alınır. Şayet P>SL ise 4.adıma, değilse son adıma(6.adım) gidilir.
        4.adım-Bu aşamada, 3.adımda seçilen ve en yüksek p değerine sahip değişken sistemden kaldırılır.
        5.adım-Makine öğrenmesi güncellenir ve 3.adıma geri dönülür.
        6.adım-Makine öğrenmesi sonlandırılır.

    3.Yöntem: İleriye seçim(Forward Selection):
        1.Significance Level(SL) seçilir.genelde 0.05
        2.Bütün değişkenler kullanılarak bir model inşa edilir.
        3.En düşük p-value değerine sahip olan değişken ele alınır.
        4.Bu aşamada, 3.adımda seçilen değişken sabit tutularak yeni bir değişken daha seçilir ve sisteme eklenir.
        5.Makine öğrenmesi güncellenir ve 3.adıma geri dönülür, şayet en düşük p-value sahip değişken için P<SL şartı sağlanıyorsa 3.adıma dönülür.Sağlanmıyorsa 6.adıma geçilir.
        6.Makine Öğrenmesi sonlandırılır.
    4.Yöntem: Çift Yönlü Eleme(Bidirectional Elimination):
        1.Significance Level(SL) seçilir.genelde 0.05
        2.Bütün değişkenler kullanılarak bir model inşa edilir.
        3.En düşük p-value sahip olan değişken ele alınır.
        4.Bu aşamada, 3.adımda seçilen değişken sabit tutularak, diğer bütün değişkenler sisteme dahil edilir ve en düşük p value sahip olan sistemde kalır.
        5.SL değerinin altında olan değişkenler sistemde kalır ve eski değişkenlerden hiçbirisi sistemden çıkarılamaz.
        6.Makine öğrenmesi sonlandırılır.
    5.Yöntem: Skor Karşılaştırması(Score Comparison):
        1. Başarı Kriteri Belirlenir.
        2.Bütün olası makine öğrenmesi modelleri inşa edilir.(ikili seçim olabilir.)
        3.Başta belirlenen kriteri(1.adım) en iyi sağlayan yöntem seçilir.
        4.Makine öğrenmesi sonlandırılır.

"""

## Lineer Regresyon
# Kütüphaneleri içe aktarma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#verileri okuma
veriler = pd.read_csv("satislar.csv")

#verileri hazırlayalım
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

#test için ayıralım
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

#veri ölçeklendirme
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

#model inşa etme
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)
x_train = pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)

x_train=x_train.sort_index()
y_train=y_train.sort_index()

#görselleştirme
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,lr.predict(x_test),color='blue')
plt.title("Aylara Göre Satış")
plt.show()

## Çoklu Regresyon

#veri okuma
veriler = pd.read_csv("veriler.csv")

#kategorik->sayısal
from sklearn import preprocessing
Yas=veriler.iloc[:,1:4].values
ulke = veriler.iloc[:, 0:1].values
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])

#onehotencoding
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

c = veriler.iloc[:,-1:].values
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
c=ohe.fit_transform(c).toarray()
print(c)

#dataframe birleştirme işlemleri
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
cinsiyet = veriler.iloc[:, -1].values
sonuc3 = pd.DataFrame(data=c[:,:1], index=range(22), columns=['cinsiyet'])
s1 = pd.concat([sonuc, sonuc2], axis=1)
s2 = pd.concat([s1, sonuc3], axis=1)

#çoklu regresyon

x_train, x_test, y_train, y_test = train_test_split(s1, sonuc3, test_size=0.33, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

boy = s2.iloc[:, 3:4].values
sol = s2.iloc[:, :3]
sag = s2.iloc[:, 4:]
veri = pd.concat([sol, sag], axis=1)
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)
regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)
y_pred = regressor2.predict(x_test)

#Geriye ELeme Yöntemi
import statsmodels.api as sm

X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
x_l=veri.iloc[:,[0,1,2,3]].values
x_l=np.array(x_l,dtype=float)
model=sm.OLS(boy,x_l).fit()
print(model.summary())

### Örnek yapalım ###

veriler = pd.read_csv("odev_tenis.csv")

#encoder: Kategorik->Numeric
#label encoder
"""
play=veriler.iloc[:,1:].values
play[:,-1]=le.fit_transform(veriler.iloc[:,-1])

print(play)
#------
windy=veriler.iloc[:,-2:-1].values
windy[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(windy)
"""

#kolay yöntem
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

#one hot encoding işlemi
c = veriler2.iloc[:, :1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
havadurumu = pd.DataFrame(data=c, index=range(14), columns=['o', 'r', 's'])
sonveriler = pd.concat([havadurumu, veriler.iloc[:, 1:3]], axis=1)
sonveriler = pd.concat([veriler2.iloc[:, -2:], sonveriler], axis=1)

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:, :-1], sonveriler.iloc[:, -1:], test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)
x_l = sonveriler.iloc[:, [0, 1, 2, 3, 4, 5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:, -1:], x_l).fit()

print(model.summary())

#gereksiz sütunları kaldıralım
sonveriler = sonveriler.iloc[:, 1:]
X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)
x_l = sonveriler.iloc[:, [0, 1, 2, 3, 4]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:, -1:], x_l).fit()

print(model.summary())

#tekrar analiz yapalım.
x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

##Polinomal Regresyon
from sklearn.preprocessing import PolynomialFeatures
veriler = pd.read_csv("maaslar.csv")
#veri dilimleme(slice)
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]

#önce lineer regresyon uygulayalım.

lineer_regression = LinearRegression()
lineer_regression.fit(x.values, y.values)
#görselleştirelim
plt.scatter(x.values, y.values, color='red')
plt.plot(x, lineer_regression.predict(x.values), color='blue')
plt.title('Doğrusal Regresyon')
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Maaş')
plt.show()

#şimdi polinomal regresyon için model oluşturalım
poly_regression = PolynomialFeatures(degree=2)
x_poly = poly_regression.fit_transform(x.values)

lineer_regression2 = LinearRegression()
lineer_regression2.fit(x_poly, y)

#görselleştirelim

plt.scatter(x.values, y.values, color='red')
plt.plot(x.values, lineer_regression2.predict(poly_regression.fit_transform(x.values)), color='blue')
plt.title('Polinomal Regresyon (Derece 2)')
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Maaş')
plt.show()


#Destek Vektör Regresyon
#veriyi yükleyelim.
veriler=pd.read_csv("maaslar.csv")

#dataframe dilimleme (slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(x.values)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(y.values)

##Support Vektor Regression import
from sklearn.svm import SVR 
svr_reg=SVR(kernel='rbf')

svr_reg.fit(x_olcekli,y_olcekli)

#görselleştirelim

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()

## Karar Ağacı

# Karar Ağacı Regresyonu import
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x.values, y.values)

#görselleştirelim.

plt.scatter(x.values, y.values, color='red')
plt.plot(x.values, dt_reg.predict(x.values), color='blue')
plt.title("Decision Tree")
plt.xlabel("Deneyim(Yıl)")
plt.ylabel("Maaş")
plt.show()


## Rassal Ağaçlar
#veri yükleme

veriler = pd.read_csv("maaslar.csv")

#slice

x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]

from sklearn.preprocessing import StandardScaler
#veri ölçeklendirme
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x.values)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y.values)
#model oluşturma
from sklearn.ensemble import RandomForestRegressor
rf_regression = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regression.fit(x.values, y.values)

#görselleştirme

plt.scatter(x.values, y.values, color='red')
plt.plot(x.values, rf_regression.predict(x.values), color='blue')
plt.xlabel("Deneyim(Yıl)")
plt.ylabel("Maaş")
plt.title("Random Forest ")
plt.show()

### BİR UYGULAMA DAHA YAPALIM ###
"""
Veri Kümesini İndirelim, Gerekli/Gereksiz Bağımsız Değişkenleri Bulalım.
5 Farklı Yönteme Göre Regresyon Modellerini Çıkaralım.
Yöntemlerin başarılarını karşılaştıralım.

"""

#veri yükleme
veriler = pd.read_csv("yenimaaslar.csv")

#veri dilimleme
x = veriler.iloc[:, 2:3]
y = veriler.iloc[:, 5:]

#korelasyon
#print(veriler.corr())

#### Modelleri kullanarak r^2 hesaplayalım ####

##Lineer regresyon modelini kullanalım.
lineer_regression=LinearRegression()
lineer_regression.fit(x.values,y.values)
model1=sm.OLS(lineer_regression.predict(x.values),x.values)
print(model1.fit().summary())

##Polinomal Regresyon##

polynomial_reg=PolynomialFeatures(degree=2)
x_poly=polynomial_reg.fit_transform(x.values)

lineer_regression2=LinearRegression()
lineer_regression2.fit(x_poly,y.values)
plt.scatter(x.values, y.values,color='red')
plt.plot(x.values,lineer_regression2.predict(x_poly),color='blue')
plt.show()
model2=sm.OLS(lineer_regression2.predict(polynomial_reg.fit_transform(x.values)),x.values)
print(model2.fit().summary())

#Veri ölçekleme
scaler1=StandardScaler()
x_olcekli=scaler1.fit_transform(x.values)
scaler2=StandardScaler()
y_olcekli=scaler2.fit_transform(y.values)

## Destek Vektör Regresyon ##
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

## Karar Ağacı ##

from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color='red')
plt.plot(x.values,dt_reg.predict(x.values),color='blue')
plt.show()
model4=sm.OLS(dt_reg.predict(x.values),x.values)
print(model4.fit().summary())

## Rassal Ağaçlar ##
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=3,random_state=0)
rf_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color='red')
plt.plot(x.values,rf_reg.predict(x.values),color='blue')
model5=sm.OLS(rf_reg.predict(x.values),x.values)
print(model5.fit().summary())


"""
Çözüm
Gerekli: Unvanseviye, Kıdem, Puan, Maas
Gereksiz: Unvan, ID
Tek Parametreli Olarak:

Linear Regression R-Squared: 0.942
Polynomial Regression R-Squared: 0.810
Support Vector Regression R-Squared: 0.770
Decision Tree R-Squared: 0.751
Random Forest R-Squared: 0.721
Üç Parametreli Olarak:

Linear Regression R-Squared: 0.903
Polynomial Regression R-Squared: 0.680
Support Vector Regression R-Squared: 0.782
Decision Tree R-Squared: 0.679
Random Forest R-Squared: 0.713



"""

#--------

"""
Linear Regression- Artılar: Veri Boyundan Bağımsız olarak doğrusal ilişki üzerine kuruludur
                   Eksiler: Doğrusallık kabulü aynı zamanda hatadır.
                   
Polynomial Regression- Artılar: Doğrusal olmayan problemleri adresler.
                       Eksiler: Başarı için doğru polinom derecesi önemlidir.
                       
Support Vektor Regression- Artılar: Doğrusal olmayan modellerde çalışır, marjinal değerlere karşı ölçekleme ile dayanıklı olur.

                          Eksiler: Ölçekleme önemlidir, anlaşılması nispeten karışıktır, doğru kernel fonksiyonu seçimi önemlidir.
                          
Decision Tree Regression- Artılar: Anlaşılabilirdir, ölçekleme ihtiyaç duymaz. Doğrusal veya doğrusal olmayan problemlerde çalışır.
                          Eksiler: Sonuçlar sabitlenmiştir. küçük veri kümelerinde ezberleme olması yüksek ihtimaldir.

Random Forest- Artılar: Ölçeklemeye ihtiyaç duymaz. Doğrusal veya doğrusal olmayan problemlerde çalışır. ezber ver sabit sonuç riski düşüktür.
               Eksiler: Çıktıların yorumu ve görsellemesi nispeten zordur.

"""



