import torch #tensor islemleri
import torch.nn as nn #yapay sinir agi katmanlarini tanimlamak icin
import torch.optim as optim #optimizasyon algoritmalari
import torchvision #goruntu isleme pre-defined modelleri
import torchvision.transforms as transforms #goruntu donusumleri
from torch.utils.data import Subset
import matplotlib.pyplot as plt #gorsellestirme
from metrics_report import save_classification_report

#device belirle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_loader(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
  
#mnist dataseti indir ve train /test kumelerini olustur
    train_set = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)


# sadece ilk 1000 örneği al
    train_subset = Subset(train_set, range(1000))
    test_subset = Subset(test_set, range(1000))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True) #trainde veriyi karistiriyoruz (shuffle)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False) #testte karıstırmıyoruz

    return train_loader, test_loader

train_loader, test_loader = get_data_loader()

#data gorsellestirme
def visualize_samples(loader,n):
    images, labels = next(iter(loader)) #ilk batch den goruntu ve etiketleri aldik
    fig, axes = plt.subplots(1, n, figsize=(10, 5)) #n farkli goruntu icin gorsellestirme alani
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off') #eksenleri gizle
    plt.show()

#visualize_samples(train_loader,9)

class NeuralNetwork(nn.Module): #pytorch un nn.module classindan miras
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() #elimizde bulunan goruntulerı (2D) vektor haline cevirelim (1D) 28*28 = 784
        self.fc1 = nn.Linear(28*28, 128) # 1. Fully Connected Layer 784 = input size , 128 = output size
        self.relu = nn.ReLU() #activation function
        self.fc2 = nn.Linear(128, 64) # 2. Fully Connected Layer : 128= input size, 64= output size
        self.fc3 = nn.Linear(64, 10) #output layer: 64=input size, 10=output size (0-9 labels)

    def forward(self, x):
        x = self.flatten(x) #initial x= 28*28 lik bit goruntu -> 784 vektor haline getir
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = NeuralNetwork().to(device)

#loss fonksiyonu ve optimizayon algoritmasini belirle
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters())
)

criterion, optimizer = define_loss_and_optimizer(model)

#train fonksiyonu

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.3f}")

        # Test loss hesapla
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {avg_test_loss:.3f}")

    #train - test graph
    plt.figure()
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_losses, marker='o', label='Training Loss',color="#FF47C8")
    plt.plot(epochs_range, test_losses, marker='s', label='Test Loss',color="#24CB61")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_loader, test_loader = get_data_loader()
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5)
    save_classification_report(model, test_loader, device)