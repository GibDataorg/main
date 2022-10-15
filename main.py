import os
import time

from Classification import Classification

import torch
import torch.utils.data
import torchvision
from pycocotools.coco import COCO
import numpy as np
import os
import albumentations.pytorch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
import re

from Visualization import drawBBox


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)
        boxes = []

        try:
            for i in range(num_objs):
                xmin = coco_annotation[i]['bbox'][0]
                ymin = coco_annotation[i]['bbox'][1]
                xmax = xmin + coco_annotation[i]['bbox'][2]
                ymax = ymin + coco_annotation[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
        except:
            pass

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {"boxes": boxes, "labels": labels, "image_id": img_id, "area": areas, "iscrowd": iscrowd}

        if self.transforms:
            augmented = self.transforms(image=np.asarray(img))
            img = augmented['image']

        return img, my_annotation, path


# хрен знает что
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# функция для DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


# функция для обучения
def teach(num_epochs):
    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)

    print("начинаем обучение на датасете длиной", len_dataloader)

    last_losses = 100
    for epoch in range(num_epochs):
        model.train()

        i = 0
        print("итерируемся по data_loader")

        for imgs, annotations, path in data_loader:

            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')

        if losses.item() < last_losses:
            last_losses = losses.item()
            torch.save(model.state_dict(), model_path)
            print("Модель сохранена")

    print("Обучение завершено")


# тест функция
def checkAndSaveTestCocoJson(submission_path, test_dir_path, threshold, data_loader):
    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='stone0'))
    coco.add_category(CocoCategory(id=1, name='stone1'))

    counter = 0
    model.eval()
    i = 0
    len_dataloader = len(data_loader)

    print("начинаем прогон на тестовом датасете длиной", len_dataloader)
    for imgs, annotations, paths in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        preds = model(imgs)
        print(f'Iteration: {i}/{len_dataloader}')

        for batch_item in preds:

            count_best_boxes = 1
            for score in batch_item['scores']:
                if score > threshold:
                    count_best_boxes += 1

            boxes = batch_item['boxes'][:count_best_boxes]
            file_name = paths[counter % train_batch_size]
            image_id = int(re.findall(r'\d+', file_name)[0])

            print("для картинки", image_id, "(", file_name, ")")

            coco_image = CocoImage(file_name=file_name, height=1080, width=1920, id=image_id)

            # ХЗ че за строки если img не используется
            img = imgs[counter % train_batch_size].cpu().detach().numpy()
            img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
#CLASSIFICATION, NEGABARIT, MAXIMUM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
                print(x_min, x_max, y_min, y_max)
                size = max(x_max-x_min, y_max-y_min)
                rec = 720 / ((y_max + y_min) / 2) * size


            try:
                drawBBox(file_name.split('.')[0], test_dir_path+file_name, boxes, 0)
                drawBBox(file_name.split('.')[0], test_dir_path + file_name, boxes, 1)
            except Exception as e:
                print("ОШИБОЧКА", e)
                quit()

            for box in boxes:
                width, height = Image.open(test_dir_path + file_name).size
                x_min = box[0].item()
                y_min = box[1].item()
                width = box[2].item() - x_min
                height = box[3].item() - y_min
                coco_image.add_annotation(
                    CocoAnnotation(
                        bbox=[x_min, y_min, width, height],
                        category_id=1,
                        category_name='stone1',
                        image_id=image_id
                    )
                )
            coco.add_image(coco_image)
            counter += 1
    save_json(data=coco.json, save_path=submission_path)
    print("Результат сохранен. Тест закончен")


if __name__ == '__main__':
    get_transform = albumentations.Compose([
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    # Batch size
    train_batch_size = 4

    # Тренировочный датасет
    train_data_dir = 'content/dataset_lite/train'
    train_coco = 'content/dataset_lite/annot_local/train_annotation.json'

    my_dataset = myOwnDataset(root=train_data_dir,
                              annotation=train_coco,
                              transforms=get_transform
                              )

    data_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    # Локальный тестовый датасет
    test_data_dir = 'content/dataset_lite/train/'
    test_coco = 'content/dataset_lite/annot_local/test_annotation.json'

    my_dataset_test = myOwnDataset(root=test_data_dir,
                                   annotation=test_coco,
                                   transforms=get_transform
                                   )

    data_loader_test = torch.utils.data.DataLoader(my_dataset_test,
                                                   batch_size=train_batch_size,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)

    device = torch.device('cpu')

    num_classes = 2

    # загрузка весов из файла
    model = get_model_instance_segmentation(num_classes)
    model_path = 'content/model'

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("Модель успешно загружена")
    except:
        print("ОШИБОЧКА")
        quit()

    model.to(device)

    # ОБУЧЕНИЕ (раскомментировать чтобы обучить)
    #num_epochs = 1
    #teach(num_epochs)

    # ТЕСТ 1 (раскомментировать чтобы прогнать тест)
    threshold = 0.99
    submission_path = "content/test.json"
    dir_path = "content/dataset_lite/train/"
    checkAndSaveTestCocoJson(submission_path, dir_path, threshold, data_loader_test)
    print("Answer is: ", Classification.type_classes)
    print("Answer2 is: ", Classification.size_classes)
