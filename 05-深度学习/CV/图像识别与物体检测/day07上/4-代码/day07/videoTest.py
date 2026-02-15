import numpy as np
import cv2
from utils.tag_video import VideoTag
from nets.ssd_net import SSD300

# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     print(frame.shape)
#
#     # 在图片显示之前去做处理
#     # grb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # 画出图形
#     # 画框的颜色
#     cv2.rectangle(frame, (300, 300), (500, 400), (0, 255, 0), 3)
#     cv2.circle(frame, (380, 380), 63, (0, 0, 255), -1)
#     cv2.putText(frame, 'python', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
#
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放资源关闭窗口
# cap.release()
# cap.destroyAllWindows()

if __name__ == '__main__':

    input_shape = (300, 300, 3)
    # 数据集的配置
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                   "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                   "tvmonitor"]
    model = SSD300(input_shape, num_classes=len(class_names))
    # 加载已训练好的模型
    model.load_weights("./ckpt/pre_trained/weights_SSD300.hdf5", by_name=True)

    vt = VideoTag(model, input_shape, len(class_names))
    vt.run(0)
