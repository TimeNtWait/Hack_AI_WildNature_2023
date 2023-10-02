import os
from ultralytics import YOLO

import pandas as pd
import torch

from animal_classification import AnimalClassification, Efficientnet_b4
from animal_detection import AnimalDetection

class UnionModel:
    def __init__(self, device):
        self.device = device
        self.load_detection()
        self.load_classification()

    def load_classification(self):
        # AnimalClassification
        # Загружаем модель
        self.classific_model = Efficientnet_b4()

        self.classific_model.load_state_dict(torch.load('efficient_weights_2class.pth'))
        self.classific_model.eval()

        self.animal_сlassification = AnimalClassification(self.classific_model, self.device)

    def load_detection(self):
        # Загружаем модель
        self.detect_model = YOLO('yolov8x.pt')

        # Определяем классы которые будем детекутировать
        # animal_classes = {1: 'bicycle',  2: 'car',  3: 'motorcycle',  4: 'airplane',  5: 'bus',  6: 'train',  7: 'truck',  9: 'traffic light',  10: 'fire hydrant',  11: 'stop sign',  12: 'parking meter',  13: 'bench',  14: 'bird',  15: 'cat',  16: 'dog',  17: 'horse',  18: 'sheep',  19: 'cow',  20: 'elephant',  21: 'bear',  22: 'zebra',  23: 'giraffe',  24: 'backpack',  25: 'umbrella',  26: 'handbag',  27: 'tie',  28: 'suitcase',  29: 'frisbee',  30: 'skis',  31: 'snowboard',  32: 'sports ball',  33: 'kite',  34: 'baseball bat',  35: 'baseball glove',  36: 'skateboard',  37: 'surfboard',  38: 'tennis racket',  39: 'bottle',  40: 'wine glass',  41: 'cup',  42: 'fork',  43: 'knife',  44: 'spoon',  45: 'bowl',  46: 'banana',  47: 'apple',  48: 'sandwich',  49: 'orange',  50: 'broccoli',  51: 'carrot',  52: 'hot dog',  53: 'pizza',  54: 'donut',  55: 'cake',  56: 'chair',  57: 'couch',  58: 'potted plant',  59: 'bed',  60: 'dining table',  61: 'toilet',  62: 'tv',  63: 'laptop',  64: 'mouse',  65: 'remote',  66: 'keyboard',  67: 'cell phone',  68: 'microwave',  69: 'oven',  70: 'toaster',  71: 'sink',  72: 'refrigerator',  73: 'book',  74: 'clock',    76: 'scissors',  77: 'teddy bear',  78: 'hair drier',  79: 'toothbrush'}
        self.animal_classes = {14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                          21: 'bear',
                          22: 'zebra', 23: 'giraffe', 77: 'teddy bear', 29: 'frisbee', 9: 'traffic light'}

        # Указываем параметры модели
        self.model_params = {'conf': 0.15, 'iou': 0.1, 'half': False, 'augment': True, 'agnostic_nms': False,
                        'retina_masks': False}
        self.animal_detection = AnimalDetection(self.detect_model, self.device)

    def predict(self, files, verbose = False, detail_verbose= False):
        if verbose:
            print("Запскаем детекцию - ищем животных")
        # Делаем прогноз
        animals, empty = self.animal_detection.predict(files, self.animal_classes, self.model_params, verbose=detail_verbose)

        if verbose:
            print(f"animals: {animals}, empty: {empty}")

        if verbose:
            print("Запскаем классификацию - ищем битые")
        # Делаем прогноз
        bads, empty = self.animal_сlassification.predict(empty, threshold=0.55, verbose=detail_verbose)
        if verbose:
            print(f"bads: {bads}, empty: {empty}")

        return animals, empty, bads, files

    @staticmethod
    def list_files(startpath):
        result_predict = {}
        for root, dirs, files in os.walk(startpath):
            animals, empty, bads, files = union_model.predict([root + '/' + f for f in files], verbose=False,
                                                              detail_verbose=False)

            for f in files:
                clear_filename = f[len(DATASET_PATH_ANIMAL):]
                clear_filename = clear_filename.replace("\\", "/")

                is_animal = 1 if f in animals else 0
                is_bad = 1 if f in bads else 0
                is_empty = 1 if (is_animal == 0) and (is_bad == 0) else 0

                result_predict[clear_filename] = (is_bad, is_empty, is_animal)
        return result_predict


    def predict_tree(self, root_path):
        result_predict = self.list_files(root_path)

        submission_df = pd.DataFrame.from_dict(result_predict).T.reset_index().rename(
            columns={'index': 'depl_pth', 0: 'broken', 1: 'empty', 2: 'animal'}).set_index('depl_pth')

        return submission_df


if __name__ == '__main__':
    # Определяем список файлов для детекции
    DATASET_PATH_ANIMAL = 'datasets/Animal/'
    DATASET_PATH_ANIMAL = 'datasets/bad/'
    DATASET_PATH_ANIMAL = 'E:/_Datasets/train_dataset_altai/clear_imgs/'
    DATASET_PATH_ANIMAL = "E:/_Datasets/train_dataset_altai/фотоловушка новое/flхатка20160929/101EK113/"
    DATASET_PATH_ANIMAL = 'datasets/Animal/'
    DATASET_PATH_ANIMAL = 'datasets/root_test/'

    path_files = os.path.abspath(DATASET_PATH_ANIMAL)
    files = [entry.path for entry in os.scandir(path_files) if entry.is_file()]


    # Если есть GPU то устанавливаем device = 'cuda'
    device = 'cuda'
    # device = 'cpu'
    union_model = UnionModel(device)
    # animals, empty, bads, files = union_model.predict(files, verbose = False, detail_verbose = False)
    # print(len(animals), len(empty), len(bads), len(files))

    submission_df = union_model.predict_tree(DATASET_PATH_ANIMAL)
    submission_df.to_csv("submission_v13.csv", sep=";", index='depl_pth')

