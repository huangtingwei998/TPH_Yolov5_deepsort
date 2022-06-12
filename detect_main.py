import json
import time


from detector_tracker import tracker
from detector_tracker.detector import Detector


import cv2
#锁定目标相关
clickX = 0  #全局变量，获取鼠标点击的点
clickY = 0  #全局变量，获取鼠标点击的点
ID=0        #全局变量，跟踪的ID
num = 1  #全局点击变量，为0或者为1


#保存视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (960,540))

def loadjson():
    with open("config/start.json", "r") as f:
        dict = json.load(f)
        if dict['choice']=='1':
            video = int(dict['video'])
        else:
            video = dict['video']
        isdetect = int(dict['isdetect'])
        print(isdetect)
    return video,isdetect

def Locationclick(event, x, y, flags, params):
    #鼠标右键单击事件
    if event == cv2.EVENT_RBUTTONDOWN:
        global clickX
        global clickY
        global num
        clickX = x
        clickY = y
        num = 1

def loadJsonFlagID():
    with open("config/findID.json", "r") as f:
        dict = json.load(f)
        ID = dict['ID']
        flag = dict['flag']
    return ID,flag

def clickID(bboxes,clickX,clickY):
    global ID
    global num
    num -=1
    flag = False
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if x1<clickX<x2 and y1<clickY<y2:
            ID = pos_id
            flag = True
            break
    data = {
        'flag': flag,
        'ID': str(ID)
    }
    print("点击的目标是" + str(ID))
    with open('config/findID.json', 'w+') as f:
        json.dump(data, f)

#设置初始的flag和ID的值
def startSet():
    data = {
        'flag': False,
        'ID': str(-1)
    }
    with open('config/findID.json', 'w+') as f:
        json.dump(data, f)

class trackermain:

    def __int__(self):
        startSet()

    def main(self):
        # 初始化 yolov5
        detector = Detector()
        #建立TCP的连接

        # 打开视频
        #capture = cv2.VideoCapture(0)   #读取本机摄像头
        capture = cv2.VideoCapture('apple.mp4')  # 读取本丢地视频
        # capture = cv2.VideoCapture('rtsp://192.168.144.25:8554/main.264')   #读取网络摄像头
        # 打开图传摄像头的RTSP流


        #由于无人机的响应速度不够快，设置设置0.5秒一个周期
        t = time.time()

        while True:

            ok, im = capture.read()
            if not ok:
                continue
            if im is None:
                break
                # 软件主窗口不需要检测
            # 缩小尺寸，1920x1080->960x540
            im = cv2.resize(im, (960, 540))
            bboxes = detector.detect(im)

            # 如果画面中 有bbox
            if len(bboxes) > 0:

                list_bboxs = tracker.update(bboxes, im)
                global num
                if num == 1:
                    clickID(list_bboxs, clickX, clickY)
                # 画框
                ID,_ = loadJsonFlagID()
                output_image_frame,centerX,centerY = tracker.draw_bboxes(im, list_bboxs, int(ID), line_thickness=None)

                #将数据发送给无人机  0.5秒一个周期
                if centerX !=-1 and centerY !=-1 and time.time()-t>=0.3:

                    print(str(int(centerX)) + " " + str(int(centerY)))
                    t = time.time()
                pass
            else:
                # 如果画面中 没有bbox
                output_image_frame = im
            pass
            # 软件主窗口不需要检测
            cv2.imshow('tracker', output_image_frame)
            out.write(output_image_frame)
            #输入q 退出检测
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("退出检测...降落")

                break
            #响应鼠标的右键点击事件
            cv2.setMouseCallback('tracker', Locationclick)
            pass

        pass
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        self.flag = False


if __name__ == '__main__':
    trackermain = trackermain()
    trackermain.main()

