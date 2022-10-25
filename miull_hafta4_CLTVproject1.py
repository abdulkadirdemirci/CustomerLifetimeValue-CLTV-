################################
# iş problemi
################################
"""
FLO satışvepazarlamafaaliyetleriiçinroadmap belirlemekistemektedir.
Şirketinortauzunvadeliplan yapabilmesiiçinvar olan müşterilerin
gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.
"""
################################
# veri seti hikayesi
################################
"""
Veri seti Flo’dan son alışverişlerini 2020 -2021 yıllarında 
OmniChannel(hem online hem offline alışverişyapan) olarak yapan
müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

master_id              - Eşsiz müşteri numarası
order_channel          - Alışveriş yapılan platforma ait hangi kanalın kullanıldığı(Android, ios, Desktop, Mobile)
last_order_channel     - En son alışverişin yapıldığı kanal
first_order_date       - Müşterinin yaptığı ilk alışveriş tarihi
last_order_date        - Müşterinin yaptığı son alışveriş tarihi
last_order_date_online        - Müşterinin online platformda yaptığı son alışveriş tarihi
last_order_date_offline       - Müşterinin offline platformda yaptığı son alışveriş tarihi
order_num_total_ever_online   - Müşterinin online platformda yaptığı toplam alışveriş sayısı
order_num_total_ever_offline  - Müşterinin offline'da yaptığı toplam alışveriş sayısı
customer_value_total_ever_offline - Müşterinin offline alışverişlerinde ödediği toplam ücret
customer_value_total_ever_online  - Müşterinin online alışverişlerinde ödediği toplam ücret
interested_in_categories_12       - Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
"""

################################
# GÖREV1 : VERİYİ ANLAMA VE HAZIRLAMA
################################

import pandas as pd
import lifetimes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
################################
# GÖREV1 - ADIM1 : flo_data_20K.csv verisin iokuyunuz
################################
df_ = pd.read_csv("C:/Users/Abdulkadir DEMİRCİ/Desktop/2022mvkpython/veri/flo_data_20K.csv")
df = df_.copy()
df.describe((0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.99, 1)).T
df.isnull().sum()

################################
# GÖREV1 - ADIM2 : Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve
# replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.
# Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız
################################
def outlier_thresholds(df,variablename,upper_quantile,lower_quantile,limiter=1.5):
    """
    outliers(aykırı deger) lara karar verebilmek için boxplot mantığındaki whiskerslar ile belirtilen
    aykırı değerleri değil kendimiz belirlediğimiz sınırı aşan değerleri outlier olarak degerlendirmemizi
    saglayan bir fonksiyon. Temelde quantilelar 0.25 lik ve 0.75 lik dilimlerde bulunur fakat bu fonksiyon
    sayesinde biz istenilen genişlikte quantilelara sahip olabiliriz.

    :param df: üzerinde işlem yapılacak dataframe, type: dataframe
    :param variablename: değişken isimleri, type: str
    :param upper_quantile: quantile3 için yüzdelik kesim, type: int or float
    :param lower_quantile: quantile1 için yüzdelik kesim, type: int or float
    :param limiter: upper ve lower limit için kullanılacak kat sayı, type: int or float, default: 1.5
    :return: up_limit type: int or float, low_limit type: int or float
    """
    quantile1 = df[variablename].quantile(lower_quantile)
    quantile3 = df[variablename].quantile(upper_quantile)
    interquantile_range = quantile3-quantile1
    up_limit  = round(quantile3 + limiter*interquantile_range)
    low_limit = round(quantile1 - limiter*interquantile_range)
    return up_limit,low_limit
def replace_with_thresholds(df,variablename,upper_quantile,lower_quantile,limiter=1.5):
    """

    :param df: üzerinde işlem yapılacak dataframe, type: dataframe
    :param variablename: değişken isimleri, type: str
    :param upper_quantile: quantile3 için yüzdelik kesim, type: int or float
    :param lower_quantile: quantile1 için yüzdelik kesim, type: int or float
    :param limiter: upper ve lower limit için kullanılacak kat sayı, type: int or float, default: 1.5
    :return:
    """
    up_limit, low_limit = outlier_thresholds(df,variablename,upper_quantile,lower_quantile,limiter=1.5)
    df.loc[(df[variablename] > up_limit),  variablename] = up_limit
    df.loc[(df[variablename] < low_limit), variablename] = low_limit

################################
# GÖREV1 - ADIM3 :  "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online"
# değişkenlerinin aykırı değerleri varsa baskılayanız
################################
replace_with_thresholds(df, "order_num_total_ever_online", 0.99, 0.01)
replace_with_thresholds(df, "order_num_total_ever_offline", 0.99, 0.01)
replace_with_thresholds(df, "customer_value_total_ever_offline", 0.99, 0.01)
replace_with_thresholds(df, "customer_value_total_ever_online", 0.99, 0.01)
df.describe().T

################################
# GÖREV1 - ADIM4 : Omnichannel müşterilerin hem online'dan hem de offline
# platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz
################################
df.head(2)
df["over_all_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["over_all_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head(2)
################################
# GÖREV1 - ADIM5 : Değişken tiplerini inceleyiniz. Tarih ifadeeden değişkenlerin tipini date'e çeviriniz
################################
import datetime as dt
df.info()
df["first_order_date"]       =pd.to_datetime(df["first_order_date"])
df["last_order_date"]        =pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] =pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"]=pd.to_datetime(df["last_order_date_offline"])
df.info()
################################
# GÖREV2 : CLTV Veri Yapısının Oluşturulması
################################

################################
# GÖREV2 - ADIM1 : Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
################################
df["last_order_date"].max()
today = dt.datetime(2021, 6, 1)
################################
# GÖREV2 - ADIM2 : customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin
# yer aldığı yeni bir cltv dataframe'i oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak,
# recency ve tenure değerleri ise haftalık cinsten ifade edilecek
################################
df["recency_cltv_weekly"] = df.apply(lambda row: ((today - row["last_order_date"]).days)/7, axis=1)
df["T_weekly"] = df.apply(lambda row: ((today - row["first_order_date"]).days)/7, axis=1)
df["frequency"] = df["over_all_order"]
df["monetary_cltv_avg"]  = df["over_all_value"] / df["frequency"]
cltv = pd.DataFrame({"customer_id":df["master_id"],
                     "recency_cltv_weekly":df["recency_cltv_weekly"],
                     "T_weekly":df["T_weekly"],
                     "frequency":df["frequency"],
                     "monetary_cltv_avg":df["monetary_cltv_avg"]},
                    columns=["customer_id", "recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"])
cltv.head()
df.head(2)
################################
# GÖREV3 : BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’ninHesaplanması
################################

################################
# GÖREV3 - ADIM1 :  BG/NBD modelinifit ediniz.
#   a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz
#   ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
#   b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz
#   ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz
################################
bgf = BetaGeoFitter(penalizer_coef=0.001)
cltv.head(2)
bgf.fit(cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])

bgf.conditional_expected_number_of_purchases_up_to_time(3*4,
                                                        cltv["frequency"],
                                                        cltv["recency_cltv_weekly"],
                                                        cltv["T_weekly"]).sort_values(ascending=False).head(10)

cltv["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(3*4,
                                                        cltv["frequency"],
                                                        cltv["recency_cltv_weekly"],
                                                        cltv["T_weekly"])

cltv["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(6*4,
                                                        cltv["frequency"],
                                                        cltv["recency_cltv_weekly"],
                                                        cltv["T_weekly"])
cltv.head(10)


################################
# GÖREV3 - ADIM2 :  Gamma-Gamma modelini fit ediniz.
# Müşterilerin ortalama bırakacakları değeri tahminleyip
# exp_average_value olarak cltv dataframe'ine ekleyiniz
################################
ggf = GammaGammaFitter(penalizer_coef=0.001)

ggf.fit(cltv["frequency"],cltv["monetary_cltv_avg"])

ggf.conditional_expected_average_profit(cltv["frequency"],cltv["monetary_cltv_avg"]).sort_values(ascending=False).head(10)

cltv.head(2)
################################
# GÖREV3 - ADIM3 :  6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
#   a. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
################################
cltv["cltv_6_month"] = ggf.customer_lifetime_value(bgf,
                                                   cltv["frequency"],
                                                   cltv["recency_cltv_weekly"],
                                                   cltv["T_weekly"],
                                                   cltv["monetary_cltv_avg"],
                                                   time=6,
                                                   freq = "W",
                                                   discount_rate=0.01)
cltv.sort_values(by="cltv_6_month",ascending=False).head(10)
################################
# GÖREV4 : CLTV Değerine Göre Segmentlerin Oluşturulması
################################

################################
# GÖREV4 - ADIM1 :   6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba(segmente)
# ayırınız ve grup isimlerini verisetine ekleyiniz.
################################
cltv["segment"] = pd.qcut(cltv["cltv_6_month"],4,labels=["D","C","B","A"])
cltv.segment.value_counts()

################################
# GÖREV4 - ADIM2 :   4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa
# 6 aylık aksiyon önerilerinde bulununuz.
################################
cltv.groupby("segment").agg({"cltv_6_month":["min","max","std","mean","count"],
                            "T_weekly":["min","max","std","mean","count"],
                            "monetary_cltv_avg":["min","max","std","mean","count"],
                            "frequency":["min","max","std","mean","count"]})
