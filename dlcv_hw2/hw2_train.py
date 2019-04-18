import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from models import Yolov1_vgg16bn
import MyDataset
import time
import visualize_bbox

def IoU(box1, box2):
    """
        box1, box2: [x, y, w, h]
        x, y is normalized to [0, 1] divided by 512/7
        w, h is normalized to [0, 1] divided by 512
    """
    w1, h1 = box1[2] * 7, box1[3] * 7
    w2, h2 = box2[2] * 7, box2[3] * 7

    area1 = w1 * h1
    area2 = w2 * h2
    
    xmin1 = box1[0] - 0.5 * w1
    xmax1 = box1[0] + 0.5 * w1
    ymin1 = box1[1] - 0.5 * h1
    ymax1 = box1[1] + 0.5 * h1
    xmin2 = box2[0] - 0.5 * w2
    xmax2 = box2[0] + 0.5 * w2
    ymin2 = box2[1] - 0.5 * h2
    ymax2 = box2[1] + 0.5 * h2

    inter_xmin = max(xmin1, xmin2) 
    inter_xmax = min(xmax1, xmax2)
    inter_ymin = max(ymin1, ymin2)
    inter_ymax = min(ymax1, ymax2)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    iou = inter / (area1 + area2 - inter)

    return iou

def loss_fn(pred, true):
    batch_size, X, Y, D = true.size()
    lambda_coord = 5
    lambda_noobj = 0.5
    
    obj_mask = true[:,:,:, 4] > 0.0
    noobj_mask = true[:,:,:, 4] == 0.0
  
    # Size: [object count, 26]
    obj_pred = pred[obj_mask]
    obj_true = true[obj_mask]
    noobj_pred = pred[noobj_mask]
    noobj_true = true[noobj_mask]

    # Classification loss
    classes_pred = obj_pred[:, 10:]
    classes_true = obj_true[:, 10:]
    class_loss = F.mse_loss(classes_pred, classes_true, reduction="sum")
    
    # No objects loss (Ground truth confidence = 0)
    confidence_pred = torch.cat((noobj_pred[:, 4], noobj_pred[:, 9]), dim=0)
    confidence_true = torch.cat((noobj_true[:, 4], noobj_true[:, 9]), dim=0)
    noobj_loss = lambda_noobj * F.mse_loss(confidence_pred, confidence_true, reduction="sum")
    
    # Objects loss (Ground truth confidence > 0)
    obj_size = len(obj_pred)
    coord_pred = obj_pred[:, :10]
    coord_true = obj_true[:, :10]
    
    better_index = torch.zeros(size=coord_true.size(), dtype=torch.uint8)
    worse_index = torch.zeros(size=coord_true.size(), dtype=torch.uint8)
    better_iou = torch.zeros(size=(coord_true.size()[0],)).cuda()
    worse_iou = torch.zeros(size=(coord_true.size()[0],)).cuda()

    for i in range(obj_size):
        bbox_1, bbox_2 = coord_pred[i, :4], coord_pred[i, 5:9]
        bbox_gt = coord_true[i, :4]
        
        iou_1 = IoU(bbox_1, bbox_gt)
        iou_2 = IoU(bbox_2, bbox_gt)

        if iou_1 > iou_2:
            better_index[i, :5] = 1
            worse_index[i, 5:10] =1
            better_iou[i] = iou_1
            worse_iou[i] = iou_2
        else:
            better_index[i, 5:10] = 1
            worse_index[i, :5] = 1
            better_iou[i] = iou_2
            worse_iou[i] = iou_1

    better_pred = coord_pred[better_index].contiguous().view(-1, 5)
    better_true = coord_true[better_index].contiguous().view(-1, 5)
    
    xy_loss = lambda_coord * F.mse_loss(better_pred[:, :2], better_true[:, :2], reduction="sum")
    wh_loss = lambda_coord * F.mse_loss(torch.sqrt(better_pred[:, 2:4]), torch.sqrt(better_true[:, 2:4]), reduction="sum")
    obj_loss = F.mse_loss(better_pred[:, 4] * better_iou, better_true[:, 4] * better_iou, reduction="sum")


    worse_pred = coord_pred[worse_index].contiguous().view(-1, 5)
    worse_true = coord_true[worse_index].contiguous().view(-1, 5)
    worse_true[:, 4] = 0

    noobj_loss += lambda_noobj * F.mse_loss(worse_pred[:, 4] * worse_iou, worse_true[:, 4], reduction="sum")
    
    return (class_loss + noobj_loss + xy_loss + wh_loss + obj_loss) / batch_size

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
        pred: [7, 7, 26]
    """
    with torch.no_grad():
        Y, X, D = pred.size()
        max_class_score, class_index = torch.max(pred[:,:, 10:], dim=2)
        
        confidence1 = pred[:,:, 4]
        confidence2 = pred[:,:, 9]

        # Size: [7, 7, 2]
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
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(448),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MyDataset.ImageDataset(image_path="./hw2-hhccode/hw2_train_val/train15000/images", label_path="./hw2-hhccode/hw2_train_val/train15000/labelTxt_hbb", transform=transform)
    train_dataset = train_dataset.__add__(MyDataset.ImageDataset(image_path="./hw2-hhccode/hw2_train_val/train15000/images", label_path="./hw2-hhccode/hw2_train_val/train15000/labelTxt_hbb", transform=transform, horizontal_flip=True))
    train_dataset = train_dataset.__add__(MyDataset.ImageDataset(image_path="./hw2-hhccode/hw2_train_val/train15000/images", label_path="./hw2-hhccode/hw2_train_val/train15000/labelTxt_hbb", transform=transform, vertical_flip=True))
    train_dataset = train_dataset.__add__(MyDataset.ImageDataset(image_path="./hw2-hhccode/hw2_train_val/train15000/images", label_path="./hw2-hhccode/hw2_train_val/train15000/labelTxt_hbb", transform=transform, horizontal_flip=True, vertical_flip=True))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=32)
    train_len = len(train_loader)

    val_dataset = MyDataset.ImageDataset(image_path="./hw2-hhccode/hw2_train_val/val1500/images", label_path="./hw2-hhccode/hw2_train_val/val1500/labelTxt_hbb", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    val_len = len(val_loader)
    
    model = Yolov1_vgg16bn(pretrained=True)
    model.to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-4)

    EPOCH = 10

    for epoch in range(EPOCH):
        model.train()

        train_loss = 0.0
        start = time.time()

        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            
            print("\r===== batch {}, loss = {} =====".format(i, loss.item()), end="")
            train_loss += loss.item()
            
        print("\n\n\nepoch {}, avg train loss = {}, time = {} seconds".format(epoch, train_loss / train_len, time.time()-start))
            
        model.eval()

        val_loss = 0.0
        start = time.time()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)


                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                

                val_loss += loss.item()
                
        print("epoch {}, avg val loss = {}, time = {} seconds\n\n".format(epoch, val_loss / val_len, time.time()-start))
       
        val_idx = [76, 86, 907]
        
        with torch.no_grad():
            for idx in val_idx:
                inputs, _ = val_dataset[idx]
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(0)

                outputs = model(inputs)

                coords, classes, scores = inference(outputs.squeeze(0))
            
                if coords is not None:
                    num = len(coords)
                else:
                    num = None

                fname = val_dataset.file_name[idx].split(".")[0]
                with open("./val_img/labels/epoch{}-{}.txt".format(epoch, fname), "w") as f:
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
                

                visualize_bbox.draw("./hw2_train_val/val1500/images/{}.jpg".format(fname), "./val_img/labels/epoch{}-{}.txt".format(epoch, fname), "./val_img/results/epoch{}-{}.jpg".format(epoch, fname))
                
                
        torch.save(model, "./ckpt/epoch{}-train_loss{:.5f}-val_loss{:.5f}.pkl".format(epoch, train_loss/train_len, val_loss/val_len))