import io
import requests
from PIL import Image
from ultralytics import YOLO  #
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import datetime

# Токен для доступа к Яндекс.Диску
TOKEN = 'y0_AgAAAAAAQVLSAAqT0AAAAADt5fzhFUEOT-TcQ2K1w-h9kS1laRFmMaE'
HEADERS = {'Authorization': f'OAuth {TOKEN}'}

# Параметры для YOLO
animal_classes = {14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                  19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'}

model_params = {'conf': 0.05, 'iou': 0.1, 'half': False, 'augment': True,
                'agnostic_nms': False, 'retina_masks': False}

# Класс для обработки изображений с YOLO
class AnimalDetection:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, image):
        try:
            y_hat = self.model.predict(image, classes=list(animal_classes.keys()),
                                       device=self.device, verbose=False, **model_params)
            classes = y_hat[0].boxes.cls.cpu().numpy()
            return len(classes) > 0
        except Exception as e:
            print(f"Error predicting image: {str(e)}")
            return False

    def move_image(self, from_path, to_path):
        move_response = requests.post('https://cloud-api.yandex.net/v1/disk/resources/move',
                                      params={'from': from_path, 'path': to_path},
                                      headers=HEADERS)
        if move_response.status_code != 201:
            print(f"Error moving the image: {move_response.text}")

    def process_folder(self, path):
        folder_content = requests.get('https://cloud-api.yandex.net/v1/disk/resources',
                                      params={'path': path, 'limit': 1000},
                                      headers=HEADERS)
        if folder_content.status_code == 200:
            items = folder_content.json().get('_embedded', {}).get('items', [])
            for item in tqdm(items):
                if item['type'] == 'file' and item.get('media_type') == 'image':
                    response = requests.get(item['file'], stream=True)
                    if response.status_code == 200:
                        image_data = io.BytesIO(response.content)

                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        new_name = f"{timestamp}_{item['name']}"

                        image = Image.open(image_data)
                        if self.predict(image):
                            self.move_image(item['path'], f"disk:/Сортированное/Животные/{new_name}")
                            self.copy_image(f"disk:/Сортированное/Животные/{new_name}", f"disk:/Архив/{new_name}")
                        else:
                            self.move_image(item['path'], f"disk:/Архив/{new_name}")
                            self.copy_image(f"disk:/Архив/{new_name}", f"disk:/Временная_папка/{new_name}")

    def copy_image(self, from_path, to_path):
        copy_response = requests.post('https://cloud-api.yandex.net/v1/disk/resources/copy',
                                      params={'from': from_path, 'path': to_path},
                                      headers=HEADERS)
        if copy_response.status_code == 201:
            print(f"Изображение скопировано.")
        else:
            print(f"Не удалось скопировать изображение. Ошибка: {copy_response.text}")



# Модель Efficientnet_b4 для дополнительной классификации
class Efficientnet_b4(nn.Module):
    def __init__(self, num_classes=2):
        super(Efficientnet_b4, self).__init__()
        self.efficientnet = models.efficientnet_b4(pretrained=False)
        self.linear_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.linear_layer(x)
        return x

# Класс для классификации изображений с EfficientNet и перемещения на ЯД
class AnimalClassification:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_folder(self, path):
        folder_content = requests.get('https://cloud-api.yandex.net/v1/disk/resources',
                                      params={'path': path, 'limit': 1000},
                                      headers=HEADERS)
        if folder_content.status_code == 200:
            items = folder_content.json().get('_embedded', {}).get('items', [])
            for item in tqdm(items):
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                new_name = f"{timestamp}_{item['name']}"

                if item['type'] == 'file' and item.get('media_type') == 'image':
                    try:
                        response = requests.get(item['file'], stream=True)
                        if response.status_code == 200:
                            image_data = io.BytesIO(response.content)
                            image = Image.open(image_data)
                            images = self.data_transform(image).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                outputs = self.model(images)
                                predicted = torch.argmax(outputs).to('cpu').item()

                            if predicted == 1:
                                # to_path = f"disk:/Сортированное/Пустые/{item['name']}"
                                to_path = f"disk:/Сортированное/Пустые/{new_name}"
                            else:
                                # to_path = f"disk:/Сортированное/Битые/{item['name']}"
                                to_path = f"disk:/Сортированное/Битые/{new_name}"

                            move_response = requests.post('https://cloud-api.yandex.net/v1/disk/resources/move',
                                                          params={'from': item['path'], 'path': to_path},
                                                          headers=HEADERS)
                            if move_response.status_code != 201:
                                print(f"Error moving the image: {move_response.text}")

                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                        # В случае ошибки, доб изображение в "Битые"
                        # to_path = f"disk:/Сортированное/Битые/{item['name']}"
                        to_path = f"disk:/Сортированное/Битые/{new_name}"
                        move_response = requests.post('https://cloud-api.yandex.net/v1/disk/resources/move',
                                                      params={'from': item['path'], 'path': to_path},
                                                      headers=HEADERS)
                        if move_response.status_code != 201:
                            print(f"Error moving the image: {move_response.text}")



if __name__ == '__main__':
    yolo_model = YOLO('yolov8x.pt')  #
    device = 'cpu'

    effnet_model = Efficientnet_b4()
    effnet_model.load_state_dict(torch.load('efficient_weights_2class.pth'))
    effnet_model.eval()

    animal_detection = AnimalDetection(yolo_model, device)
    animal_classification = AnimalClassification(effnet_model, device)

    while True:
        print("Обработка изображений с YOLO...")
        animal_detection.process_folder('disk:/Для_сортировки')

        print("Обработка изображений с EfficientNet...")
        animal_classification.process_folder('disk:/Временная_папка')

        print("Пауза перед следующей итерацией...")
        time.sleep(300)
