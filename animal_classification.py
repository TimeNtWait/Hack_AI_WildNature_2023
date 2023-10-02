
import os
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import torchvision.models as models

class AnimalClassification:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])        

    def predict(self, image_files, threshold = 0.5, verbose=False):
        empty = []
        bad = []
        self.model.to(self.device)
        for filename in tqdm(image_files):
            # Загружаем изображение
            if verbose:
                print(filename)
            try:
                image = Image.open(filename)
                images = self.data_transform(image)
            except:
                bad.append(filename)
                continue

            images = images.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                # predicted = torch.argmax(outputs).to('cpu').item()
                predicted = torch.sigmoid(outputs).to('cpu').numpy()

            if verbose:
                print(filename)

            # Применяем трешхолд
            # predicted = (predicted[:, 1] > threshold).astype(int)
            if predicted[:, 1] >= threshold:
                empty.append(filename)
            else:
                bad.append(filename)
                if verbose:
                    print(f"___bad file: {filename}")
            if verbose:
                print(f"bad: {len(bad)}, empty: {len(empty)}", )
        return bad, empty



class Efficientnet_b4(nn.Module):
    def __init__(self, num_classes=2):
        super(Efficientnet_b4, self).__init__()
        self.efficientnet = models.efficientnet_b4 ()
        self.linear_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.linear_layer(x)
        return x



if __name__ == '__main__':
    # Определяем список файлов для детекции
    # DATASET_PATH_ANIMAL = r'I:\val\1'
    DATASET_PATH_ANIMAL = r'datasets/bad/'
    # DATASET_PATH_ANIMAL = r'datasets/Animal/'
    path_files = os.path.abspath(DATASET_PATH_ANIMAL)
    files = [entry.path for entry in os.scandir(path_files) if entry.is_file()]

    # Загружаем модель
    model = Efficientnet_b4()
    model.load_state_dict(torch.load('efficient_weights_2class.pth'))
    model.eval()

    # Если есть GPU то устанавливаем device = 'cuda'
    device = f"cuda" if torch.cuda.is_available() else "cpu"

    animal_сlassification = AnimalClassification(model, device)
    # Делаем прогноз
    bad, empty = animal_сlassification.predict(files, verbose=False)

    print(len(bad), len(empty))

