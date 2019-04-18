import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from models import Yolov1_vgg16bn
import MyDataset


"""
    For improved Yolo v1 model
"""

def nms(bbox, obj, confidence):
    """
        bbox: [N, 4]
        obj: [N]
        confidence: [N]
    """
    if len(bbox) == 0:
        return None, None, None

    N, _ = bbox.size()
    threshold = 0.4

    index = torch.argsort(confidence, descending=True)
    sorted_bbox = bbox[index, :]
    sorted_obj = obj[index]
    sorted_confidence = confidence[index]

    candidates = torch.zeros(size=(N,), dtype=torch.uint8)
    
    for i in range(N):
        if sorted_confidence[i] != 0.0:
            candidates[i] = 1
        else:
            continue
        
        if i == N-1:
            break
        
        xmin1, xmax1 = sorted_bbox[i, 0], sorted_bbox[i, 1]
        ymin1, ymax1 = sorted_bbox[i, 2], sorted_bbox[i, 3]
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)

        for j in range(i+1, N):
            xmin2, xmax2 = sorted_bbox[j, 0], sorted_bbox[j, 1]
            ymin2, ymax2 = sorted_bbox[j, 2], sorted_bbox[j, 3]
            area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

            inter_xmin = max(xmin1, xmin2) 
            inter_xmax = min(xmax1, xmax2)
            inter_ymin = max(ymin1, ymin2)
            inter_ymax = min(ymax1, ymax2)

            if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
                continue
    
            inter = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

            iou = inter / (area1 + area2 - inter)
            
            if iou >= threshold:
                sorted_confidence[j] = 0.0
    
    candi_bbox = sorted_bbox[candidates]
    candi_obj = sorted_obj[candidates]
    candi_confidence = sorted_confidence[candidates]
    
    return candi_bbox, candi_obj, candi_confidence


def inference(pred):
    """
        pred: [8, 8, 26]
    """
    with torch.no_grad():
        Y, X, D = pred.size()
        max_class_score, class_index = torch.max(pred[:,:, 10:], dim=2)
        
        confidence1 = pred[:,:, 4]
        confidence2 = pred[:,:, 9]

        # Size: [8, 8, 2]
        csc_score = torch.stack((confidence1 * max_class_score, confidence2 * max_class_score), dim=2)
        mask = csc_score > 0.1

        coords = []
        classes = []
        scores = []

        for y_grid in range(Y):
            for x_grid in range(X):
                for bbox in range(2):
                    if mask[y_grid][x_grid][bbox] == 1:
                        w = pred[y_grid][x_grid][bbox*5+2] * 512
                        h = pred[y_grid][x_grid][bbox*5+3] * 512
                    
                        x = (pred[y_grid][x_grid][bbox*5] + x_grid) * 512 / X
                        y = (pred[y_grid][x_grid][bbox*5+1] + y_grid) * 512 / Y
                    
                        xmin = torch.round(x - 0.5 * w)
                        xmax = torch.round(x + 0.5 * w)
                        ymin = torch.round(y - 0.5 * h)
                        ymax = torch.round(y + 0.5 * h)

                        if not 1 <= xmin <= 512 or not 1 <= xmax <= 512 or not 1 <= ymin <= 512 or not 1 <= ymax <= 512:
                            continue
                        
                        coords.append([xmin, xmax, ymin, ymax])
                        classes.append(class_index[y_grid][x_grid])
                        scores.append(csc_score[y_grid][x_grid][bbox])

        coords = torch.tensor(coords, dtype=torch.int)
        classes = torch.tensor(classes, dtype=torch.uint8)
        scores = torch.tensor(scores)
        
        return nms(coords, classes, scores)

if __name__ == "__main__":
    CLASS = [
            "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
            "basketball-court", "ground-track-field", "harbor", "bridge", "small-vehicle",
            "large-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
            "container-crane"
    ]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2], exist_ok=True)

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = MyDataset.ImageDataset(image_path=sys.argv[1], transform=transform, grid_num=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_len = len(val_loader)

    model = torch.load("./hw2_best_model.pkl", map_location=device)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs = data
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            coords, classes, scores = inference(outputs.squeeze(0))
            
            if coords is not None:
                num = len(coords)
            else:
                num = None

            fname = val_dataset.file_name[i].split(".")[0]
            with open(os.path.join(sys.argv[2], "{}.txt".format(fname)), "w") as f:
                if num is None:
                    continue
                for j in range(num):
                    xmin, xmax = coords[j, 0], coords[j, 1]
                    ymin, ymax = coords[j, 2], coords[j, 3]
                    
                    print("{} {} {} {} {} {} {} {} {} {}".format(
                                                                xmin, ymin,
                                                                xmax, ymin,
                                                                xmax, ymax,
                                                                xmin, ymax,
                                                                CLASS[classes[j].item()],
                                                                scores[j]
                                                            ), file=f)

            print("\r Finish file {}".format(i), end="")
            