# Fine-Tuning Documentation

Bu doküman, Hugging Face Transformers kütüphanesi kullanılarak bir pretrained modelin özel bir veri seti üzerinde **fine-tuning**  nasıl yapılır adım adım anlatmaktadır.

---

## Fine-Tuning Nedir?

Fine-tuning, genel amaçlı büyük veri üzerinde eğitilmiş bir modelin, daha küçük ve özel bir veri seti üzerinde yeniden eğitilerek belirli bir göreve uyarlanmasıdır. Bu sayede model, hedef görevde daha başarılı olur.

---

## Gerekli Kütüphaneler

```bash
pip install transformers datasets evaluate accelerate torch
```

---

## 1. Veri Seti (+Ön işleme) ve Tokenizer & Model Yükleme

Aşağıda, “emotion” veri seti yüklenip tokenize edilip, eğitim ve doğrulama için küçük alt kümeler seçiliyor. Aynı zamanda bert-base-cased modeli ve tokenizer’ı tek seferde yükleniyor. Bu örnek, fine-tuning için veri hazırlama ve model yükleme aşamasını gösterir:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-cased"

dataset = load_dataset("emotion")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)  # num_labels dataset'e göre değişir, 'emotion'da 6 label var.

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

small_train = dataset["train"].shuffle(seed=42).select(range(1000))
small_eval = dataset["validation"].shuffle(seed=42).select(range(500))
```

---

Bu adımda, modelin anlayabileceği formata dönüştürmek için ham metin verileri **tokenize** edilir.

- `preprocess_function` fonksiyonu, veri setindeki her metin örneğini alır ve `tokenizer` kullanarak kelimeleri veya alt kelimeleri (token) modelin kabul ettiği sayısal temsilcilerine çevirir.
- `truncation=True` parametresi, çok uzun metinlerin modelin maksimum kabul ettiği uzunluğa göre kısaltılmasını sağlar.
- `dataset.map()` fonksiyonu, tüm veri seti örneklerine bu tokenizasyon işlemini topluca uygular ve `tokenized_datasets` olarak döner.

Bu sayede model eğitimi için veriler hazır hale gelir.

```python
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
```

---

## 2. Değerlendirme Metriklerinin Tanımlanması

Bu bölümde, modelin performansını ölçmek için bir **değerlendirme metriği** tanımlanır.

- `evaluate` kütüphanesi kullanılarak `accuracy` (doğruluk) metriği yüklenir.
- `compute_metrics` fonksiyonu, modelin tahminleri (`logits`) ve gerçek etiketleri (`labels`) alır.
- `logits.argmax(axis=-1)` ifadesi, her örnek için en yüksek olasılığa sahip sınıfı tahmin olarak seçer.
- Daha sonra, tahminler ve gerçek etiketler karşılaştırılarak doğruluk değeri hesaplanır.

Bu fonksiyon, Hugging Face `Trainer` tarafından eğitim ve değerlendirme sırasında otomatik olarak çağrılır ve modelin ne kadar iyi öğrendiğini sayısal olarak gösterir.

```python
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)
```

---

## 3. Train Ayarları

```python
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="emotion_classifier",        # train sonrası model dosyalarının kaydedileceği klasör
    eval_strategy="no",                  # validation her epoch sonunda yapılacak
    save_strategy="no",                  # model kaydı her epoch sonunda yapılacak
    learning_rate=2e-5,                     # ogrenme hızı
    per_device_train_batch_size=8,          # her train adımında kullanılacak batch boyutu 
    per_device_eval_batch_size=8,           # değerlendirme sırasında kullanılacak batch boyutu (eval)
    num_train_epochs=1,                     # Toplam 1 epoch
    weight_decay=0.01,                      # ağırlık cürümesi (regularization için)
    push_to_hub=False,                      #cok yavas calıstı false yaptim, egitilen modeli hub a yüklemek icin.
    logging_steps=50,                       #her 50 adimda loglama yap
    fp16=True,                              # 16-bit ile hız ve bellek tasarrufu sağlar (benimki yavas calistigi icin kullandim)
    report_to="none"                        
)
```

---

## 4. Trainer Nesnesinin Oluşturulması
Bu adımda, Hugging Face’in Trainer sınıfı kullanılarak model eğitimi için gerekli tüm bileşenler bir araya getirilir:

- `model`: Eğitilecek önceden yüklenmiş model.

- `args`: Eğitim ayarlarını içeren TrainingArguments nesnesi.

- `train_dataset ve eval_dataset`: Eğitim ve doğrulama için hazırlanmış veri kümeleri.

- `tokenizer`: Metinleri modele uygun hale getiren tokenizer.

- `compute_metrics`: Eğitim ve değerlendirme sırasında model performansını ölçmek için tanımlanmış metrik fonksiyonu.

Trainer nesnesi, modelin eğitimi, değerlendirilmesi ve kaydedilmesi gibi işlemleri kolaylaştırır.
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

---

## 5 Model Eğitiminin Başlatılması

```python
trainer.train()
```

---

## 8. Eğitilen Modelin Kaydedilmesi

```python
trainer.save_model("final_finetuned_model")
tokenizer.save_pretrained("final_finetuned_model")
```

---

## 9. Eğitilmiş Model ile Tahmin

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="final_finetuned_model")
print(classifier("I am so happy today!"))
```

---

##

- Eğitim süresi: 43.m 22.7s - bu kısımda ilk denememde 81m sürdü, kodda birkaç değişiklik yapmak zorunda kaldım (batch, epoch gibi)

---
