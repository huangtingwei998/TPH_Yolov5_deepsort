import time

import cv2
import torch
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
trackerpoint = (0,0)
point_size = 2   #控制点的半径
thickness = 2    #控制是否实心
point_color = (0, 0, 255)  # BGR （蓝, 绿, 红)
points_list = []  #点的坐标

cfg = get_config()
cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)



def draw_bboxes(image, bboxes,ID, line_thickness):
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_pts = []
    point_radius = 4
    flag, x11, y11, x21, y21 = findID(bboxes,ID)
    if flag==False:
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            color = (0, 255, 0)
            # 撞线的点
            check_point_x = x1
            check_point_y = int(y1 + ((y2 - y1) * 0.6))

            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
            cutImage(cls_id,c1,c2,image,pos_id)
            font_thickness = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                        [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

            list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
            list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

            ndarray_pts = np.array(list_pts, np.int32)

            cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

            list_pts.clear()
    else:
        c1, c2 = (x11, y11), (x21, y21)
        pointx = int((x11+x21)/2)
        pointy = int((y11+y21)/2)
        global trackerpoint
        trackerpoint=(pointx,pointy)
        # points_list.append((pointx,pointy))   #将中心点加入一个列表
        for point in points_list:
            cv2.circle(image, point, point_size, point_color, thickness)

        targetrectangle(image, c1, c2,line_thickness=None)
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            color = (0, 255, 0)
            # 撞线的点
            check_point_x = x1
            check_point_y = int(y1 + ((y2 - y1) * 0.6))

            c1, c2 = (x1, y1), (x2, y2)
            # cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

            font_thickness = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            # cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
            #             [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

            list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
            list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

            ndarray_pts = np.array(list_pts, np.int32)

            # cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

            list_pts.clear()
    if flag:
        centerX,centerY = (x11+x21)/2,(y11+y21)/2
    else:
        centerX,centerY = -1,-1
    return image,centerX,centerY

# 判断当前画面的bboxs是否具有跟踪的ID
# 返回跟踪的ID的坐标与标志位
#
def findID(bboxes,searchID):
    flag = False
    x11, y11, x21, y21 = 0,0,0,0
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if pos_id == searchID:
            x11, y11, x21, y21 = x1, y1, x2, y2
            flag = True
            break
    return flag,x11, y11, x21, y21


def targetrectangle(image,c1,c2,line_thickness):
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1
    (x11, y11), (x21, y21) = c1,c2
    witch = int((x21-x11)/4)
    higth = int((y21 -y11)/4)
    cv2.line(image, (x11, y11), (x11, y11+higth), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x11, y11), (x11+witch, y11), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x21-witch, y11), (x21, y11), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x21, y11), (x21, y11+higth), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x11, y21-higth), (x11, y21), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x11, y21), (x11+witch, y21), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x21, y21-higth), (x21, y21), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x21-witch, y21), (x21, y21), (0, 255, 0),thickness=line_thickness, lineType=cv2.LINE_AA)

def update(bboxes, image):
    bbox_xywh = []
    confs = []
    bboxes2draw = []

    if len(bboxes) > 0:
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),
                x2 - x1, y2 - y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, image)

        for x1, y1, x2, y2, track_id in list(outputs):
            # x1, y1, x2, y2, track_id = value
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            label = search_label(center_x=center_x, center_y=center_y,
                                 bboxes_xyxy=bboxes, max_dist_threshold=20.0)

            bboxes2draw.append((x1, y1, x2, y2, label, track_id))
        pass
    pass

    return bboxes2draw


def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在 yolov5 的 bbox 中搜索中心点最接近的label
    :param center_x:
    :param center_y:
    :param bboxes_xyxy:
    :param max_dist_threshold:
    :return: 字符串
    """
    label = ''
    # min_label = ''
    min_dist = -1.0

    for x1, y1, x2, y2, lbl, conf in bboxes_xyxy:
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 横纵距离都小于 max_dist
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)

        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 距离阈值，判断是否在允许误差范围内
            # 取 x, y 方向上的距离平均值
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次赋值
                min_dist = avg_dist
                # 赋值label
                label = lbl
                pass
            else:
                # 若不是第一次，则距离小的优先
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    # label
                    label = lbl
                pass
            pass
        pass

    return label


def gettrackerpoint():
    global trackerpoint
    return trackerpoint


def cutImage(lbl,c1,c2,im,id):
    x1,y1 = c1
    x2,y2 = c2
    if lbl in ['person']:
        img = im[y1:y2, x1:x2]
        filepath = "dataset/person/" + "id" +"_"+str(id)+ "_" + str(int(time.time() / 2)) + ".jpg"
        cv2.imwrite(filepath, img)