dataset.py ' yi çalıştırdığımda çıktı aşağıdaki gibi olmaktadır.
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   label    5572 non-null   object
 1   message  5572 non-null   object
dtypes: object(2)
memory usage: 87.2+ KB
Number of missing values per column:
label      0
message    0
dtype: int64
Number of duplicate records: 403
Record count after removing duplicates: 5169
label
ham     4516
spam     653
Name: count, dtype: int64

![alt text](image.png)


Veri seti 5572 SMS mesajından oluşmakta ve iki sütun var: label ve message. Eksik değer kontrolü yaptığımda label ve message' da eksik kayıt bulunmadı. Mesajların 4825’i ham, 747’si ise spam olarak buldum. Veri setinde 403 adet tamamen aynı olan tekrar eden kayıtlar tespit edildi  ve bu kayıtlar temizlenerek geriye 5169 farklı veri kaldı. Temizlenmiş veri sms_clean.csv dosyasına kaydedildi.

