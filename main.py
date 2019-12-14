# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:58:31 2019

@author: Toprak
"""

# Ödevdeki kısımlar 180. Satırda başlıyor

# Çizimler, grafikler için gerekli kütüphaneler

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Sigmoid fonksiyonu ve türevi

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Yakınsanmak istenen fonksiyon

def func(x1, x2):
    return x1**2 + x2**2

#def sigmoid(x):
#    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#    return t
#
#def sigmoid_der(x):
#    return 1-sigmoid(x)**2



# Yapay Sinir Ağı 

# x girdiler 
# z çıktılar
# lr öğrenme oranı
# gk1 1. gizli katman nöron sayısı 
# gk2 2. gizli katmandaki nöron sayısı
# iterSay algoritmanın kaç epoch çalışacağı
# hataOran algoritmanın hangi hata oranında duracağı

def YSA(x,z,lr,gk1,gk2,iterSay,hataOran):
    
    gizlikatman1 = gk1
    gizlikatman2 = gk2
    inputsayısı = len(x[0])
    
    # Ağırlıkların random olarak oluşturulması
    
    w1 = np.random.rand(gizlikatman1,inputsayısı)
    w2 = np.random.rand(gizlikatman2,gizlikatman1)
    w3 = np.random.rand(1,gizlikatman2)
    
    bw0 = np.random.rand(1,2)
    bw1 = np.random.rand(1,gk1)
    bw2 = np.random.rand(1,gk2)
    bw3 = np.random.rand(1,gk2)
    
    # w1 /= 1000
    # w2 /= 1000
    # w3 /= 1000
    
    # bw0 /= 1000
    # bw1 /= 1000
    # bw2 /= 1000
    # bw3 /= 1000
    
    # Hataların grafiğinin seçilmesi için
    
    toplamciz = []
    karetoplamciz = []
    
    # k iterasyon sayısını tutmak için
    # karetoplam hata oranı while başlangıçı için ilk değer
    
    karetoplam = 10**20
    k = 0
    pHata = 2
    hata = 1
    
#    for k in range(iterSay):
    while (k < iterSay and (karetoplam / n) > hataOran and pHata > (hata)):
        
        # Kaçıncı iterasyonda olunduğunun yazdırılması
        
        pHata = (karetoplam / n)
        
        print("-----------------------------------------------------------" + str(k))      
        toplam = 0
        karetoplam = 0
        for i in range(n):
            
            # katman katman sonuçların hesaplanması
            
            h = np.dot(x[i]+bw0,w1.T).reshape(1,gizlikatman1)
            
            h += bw1
            
            m = np.dot(h,w2.T).reshape(1,gizlikatman2)
            
            ms = sigmoid(m + bw3)
            
            ms += bw2
            
            output = np.dot(ms,w3.T).reshape(1,1)
            
            delta = z[i] - output
            
            # Türevlerin hesaplanması
            
            derw3 = sigmoid_der(m) * delta * lr       
            
            derw2 = np.dot((w3 * sigmoid_der(m)).T,h) * delta * lr      
            
            derw1 = np.dot(np.dot(w3 * sigmoid_der(m),w2).T,x[i].reshape(2,1).T) * delta * lr
            
            # Ağırlıkların güncellenmesi
            
            w3 += derw3
            
            w2 += derw2
            
            w1 += derw1
            
            bw0 += delta * lr
            bw1 += delta * lr
            bw2 += delta * lr
            bw3 += delta * lr
            
            # Hataların toplanması
            
            toplam += abs(delta)
            karetoplam += (delta**2)
        
        # Hata oranının yazdırılması ve listeye kaydedilmesi
        
        k += 1
        toplamciz.append(karetoplam[0][0] / n)
        print("Hata Oranı " + str(karetoplam / n))
        karetoplamciz.append(float(karetoplam / n))
        hata = (karetoplam / n)
    
    if pHata < (karetoplam / n):
        print("Local minimumda takıldı. İstenilen iterasyon veya hata oranından önce çıkıldı")
        print("Parametereleri değiştirerek deneyebilirsiniz lr = 0.00005 veya daha düşük olmalı")
    elif k > iterSay:
        print("Maksimum iterasyon sayısı aşıldı")
    elif ((karetoplam / n) < hataOran):
        print("istenilen hata oranına ulaşıldı ve çıkıldı.")
    
    
    return bw0,bw1,bw2,bw3,w1,w2,w3,karetoplamciz

# Yapay sinir ağının bulduğu ağırlık değerleriyle tahmin yapan fonksiyon

def tahmin(bw0,bw1,bw2,bw3,w1,w2,w3,sayı):
    h = np.dot(sayı+bw0,w1.T).reshape(1,len(w1))
    
    h += bw1
    
    m = np.dot(h,w2.T).reshape(1,len(w2))
    
    ms = sigmoid(m + bw3)
    
    ms += bw2
    
    output = np.dot(ms,w3.T).reshape(1,1)
    
#    print(output)
    
    return output

############################## 1. Kısım ##############################

# -2 2 aralığında 100 adet yapay veri seti oluşturup karıştırılması

n = 100
x = np.linspace(-2, 2, num=n).reshape(n,1)
y = np.linspace(-2, 2, num=n).reshape(n,1)

np.random.shuffle(x)
np.random.shuffle(y)

Z_Train = np.zeros((n,))

XY_Train = np.concatenate([x,y],axis = 1)

np.random.shuffle(XY_Train)

# Çıktı fonksiyonunun hazırlanması

for i in range(n):
    Z_Train[i] = func(XY_Train[i,0], XY_Train[i,1])


############################## 2. Kısım ##############################
    
# Fonksiyonun 3 boyutlu olarak yazdırılması

x = np.linspace(-2, 2, num=n).reshape(n,1)
y = np.linspace(-2, 2, num=n).reshape(n,1)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)


fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.title.set_text('Veri setinin 3 boyutlu olarak çizdirilmesi')

###################### 3. Kısım ##############################

# Yapay Sinir Ağının Fonksiyonunun çağırılması 0.00005

bw0,bw1,bw2,bw3,w1,w2,w3,hataList = YSA(XY_Train, Z_Train, lr = 0.00005, gk1 = 100, gk2 = 64, iterSay = 10000, hataOran = 0.05)

X, Y = np.meshgrid(x, y)
Z = func(X, Y)

Z2 = np.zeros((n, n))

# 3 Boyutlu için için Yapay Sinir Ağının ağırlıkları ile 
# X Y mesh grid verilerinin tahmininin yapılması

for i in range(n):
    for j in range(n):
        Z2[i][j] = tahmin(bw0,bw1,bw2,bw3,w1,w2,w3,[X[i][j],Y[i][j]])



# 3 Boyutlu olarak gerçek fonksiyon ile YSA tahmin değerlerinin çizdirilmesi
        
# Yeşil gerçek fonksiyon Turuncu YSA tahmini

fig = plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='green')
ax.plot_wireframe(X, Y, Z2, color='orange')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.title.set_text('Gerçek fonksiyon ile YSA Karşılaştırması')

# İterasyon sayısına göre hata oranının yazdırılması

fig1 = plt.figure(3)
plt.plot(hataList[:500])
plt.ylabel('Hata Oranı')
plt.xlabel('İterasyon Sayısı')
plt.show()
fig1.suptitle('Hata Oranı - İterasyon Sayısı')

###################### 6. Kısım ##############################

# # Aynı girdi, çıktı katman sayısı, aktivasyon fonksiyonları öğrenme oranı ile
# # Keras kütüphanesinin yapay sinir ağının eğitilmesi

# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv1D, MaxPooling1D, Reshape
# from keras.callbacks import ModelCheckpoint
# from keras.models import model_from_json
# from keras import backend as K
# from keras import optimizers

# input_shape = (2,)

# model = Sequential()
# model.add(Dense(100, activation='linear',
#                   input_shape=input_shape))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(1, activation='linear'))

# sgd = keras.optimizers.RMSprop(lr=0.0001)
# model.compile(loss=keras.losses.mean_squared_error,
#               optimizer=sgd,
#               metrics=['mae'])

# epochs = 1000
# batch_size = 1

# history = model.fit(XY_Train, Z_Train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(XY_Train, Z_Train))




# # Gerçek Fonksiyon Yazılan YSA fonksiyonu ve keras kütüphanesinin karşılaştırılması

# X, Y = np.meshgrid(x, y)
# Z = func(X, Y)

# Z3 = np.zeros((n, n))

# for i in range(n):
#     for j in range(n):
#         Z3[i][j] = model.predict(      np.array(  [[  X[i][j],Y[i][j]    ]]   )             )


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(X, Y, Z, color='green')
# ax.plot_wireframe(X, Y, Z3, color='blue')
# ax.plot_wireframe(X, Y, Z2, color='orange')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

