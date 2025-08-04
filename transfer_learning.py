import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm #egitim sürecini izlemek icin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics  import confusion_matrix , classification_report
from torchvision.models import resnet18, ResNet18_Weights
#import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)), #resnet18 inpıt size
    transforms.RandomHorizontalFlip(), #goruntuleri yatay cevirerek veri arttirimi
    transforms.RandomRotation(10), #goruntuleri rastgele 10 dereceye kadar dondur
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #renk varyasyonlari      
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #resnet18 piksel degerleri icin normalize
])

#test veri setine data augmentation yapmiyoruz, kontrollu olmasi lazim
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.Flowers102(root='./data', split='train', download=True, transform=transforms_train)
test_dataset = datasets.Flowers102(root='./data', split='val', download=True, transform=transform_test)

indicec=torch.randint(len(train_dataset),size=(5,))
samples = [train_dataset[i] for i in indicec]

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, (image, label) in enumerate(samples):
   image=image.numpy().transpose((1, 2, 0))
   image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406] #yukarıda yapilan normalizasyonun tersi (orijinali gorebilmek icin)
   #image = np.clip(image, 0, 1)  # Burada clipping yapıyoruz
   axes[i].imshow(image)
   axes[i].set_title(f"Label: {label}")
   axes[i].axis('off')
plt.show()

#dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights) # pretrained=True -> onceden egitilmis agirliklari kullan 
num_features = model.fc.in_features #mevcut sınıflandırıcının giris ozelliklerini al
model.fc = nn.Linear(num_features, 102) #fc->fully connected
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#model training
epochs = 5
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step() 
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

model.eval()
sample_image, _ = test_dataset[0]
sample_image = sample_image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(sample_image)
    print(f"Model output shape (logits): {output.shape}")

feature_extractor = nn.Sequential(*list(model.children())[:-1])
with torch.no_grad():
    embedding = feature_extractor(sample_image)
    embedding = embedding.view(embedding.size(0), -1)
    print(f"Embedding shape (before fc layer): {embedding.shape}")

torch.save(model.state_dict(), 'resnet18_flowes102_model.pth') #modeli kaydet

#model evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


report = classification_report(all_labels, all_preds,zero_division=0)
with open("classification_report.txt", "w") as f:
    f.write(report)












