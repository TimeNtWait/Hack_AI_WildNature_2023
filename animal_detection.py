from ultralytics import YOLO
import os
from tqdm import tqdm


class AnimalDetection:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, image_files, animal_classes, model_params, verbose=False):
        empty = []
        animals = []
        for filename in tqdm(image_files):
            y_hat = self.model.predict(filename, classes=list(animal_classes.keys()), device=self.device, verbose=False,
                                  **model_params)
            classes = y_hat[0].boxes.cls.cpu().numpy()
            labels = [animal_classes[i] for i in classes]

            if verbose:
                print(filename)
                print(labels)
            if len(labels) > 0:
                animals.append(filename)
            else:
                empty.append(filename)
        return animals, empty


if __name__ == '__main__':
    # Определяем список файлов для детекции
    DATASET_PATH_ANIMAL = 'datasets/Animal/'
    DATASET_PATH_ANIMAL = 'datasets/bad/'
    path_files = os.path.abspath(DATASET_PATH_ANIMAL)
    files = [entry.path for entry in os.scandir(path_files) if entry.is_file()]

    # Загружаем модель
    model = YOLO('yolov8x.pt')

    # Если есть GPU то устанавливаем device = 'cuda'
    device = 'cuda'
    # device = 'cpu'

    # Определяем классы которые будем детекутировать
    # animal_classes = {1: 'bicycle',  2: 'car',  3: 'motorcycle',  4: 'airplane',  5: 'bus',  6: 'train',  7: 'truck',  9: 'traffic light',  10: 'fire hydrant',  11: 'stop sign',  12: 'parking meter',  13: 'bench',  14: 'bird',  15: 'cat',  16: 'dog',  17: 'horse',  18: 'sheep',  19: 'cow',  20: 'elephant',  21: 'bear',  22: 'zebra',  23: 'giraffe',  24: 'backpack',  25: 'umbrella',  26: 'handbag',  27: 'tie',  28: 'suitcase',  29: 'frisbee',  30: 'skis',  31: 'snowboard',  32: 'sports ball',  33: 'kite',  34: 'baseball bat',  35: 'baseball glove',  36: 'skateboard',  37: 'surfboard',  38: 'tennis racket',  39: 'bottle',  40: 'wine glass',  41: 'cup',  42: 'fork',  43: 'knife',  44: 'spoon',  45: 'bowl',  46: 'banana',  47: 'apple',  48: 'sandwich',  49: 'orange',  50: 'broccoli',  51: 'carrot',  52: 'hot dog',  53: 'pizza',  54: 'donut',  55: 'cake',  56: 'chair',  57: 'couch',  58: 'potted plant',  59: 'bed',  60: 'dining table',  61: 'toilet',  62: 'tv',  63: 'laptop',  64: 'mouse',  65: 'remote',  66: 'keyboard',  67: 'cell phone',  68: 'microwave',  69: 'oven',  70: 'toaster',  71: 'sink',  72: 'refrigerator',  73: 'book',  74: 'clock',    76: 'scissors',  77: 'teddy bear',  78: 'hair drier',  79: 'toothbrush'}
    animal_classes = {14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                      22: 'zebra', 23: 'giraffe', 77: 'teddy bear', 29: 'frisbee', 9: 'traffic light'}

    # Указываем параметры модели
    model_params = {'conf': 0.05, 'iou': 0.1, 'half': False, 'augment': True, 'agnostic_nms': False,
                    'retina_masks': False}

    animal_detection = AnimalDetection(model, device)
    # Делаем прогноз
    animals, empty = animal_detection.predict(files, animal_classes, model_params, verbose=False)

    print(len(animals), len(empty))
