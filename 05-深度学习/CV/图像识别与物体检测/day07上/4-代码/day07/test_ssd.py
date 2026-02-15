from nets.ssd_net import SSD300
from utils.ssd_utils import BBoxUtility
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import os
"""
- 定义好类别数量以及输出
- 模型预测流程
  - SSD300模型输入以及加载参数
  - 读取多个本地路径测试图片，preprocess_input以及保存图像像素值（显示需要）
  - 模型预测结果，得到7308个priorbox
  - 进行非最大抑制算法处理
- 图片的预测结果显示

"""


class SSDTest(object):

    def __init__(self):
        # 定义识别类别
        self.classes_name = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                             'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                             'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                             'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        # 定义模型的输入参数 1北京
        self.classes_nums = len(self.classes_name) + 1
        self.input_shape = (300, 300, 3)

    def test(self):
        """
        对于输入图片进行预测物体位置
        :return:
        """
        # - SSD300模型输入以及加载参数
        model = SSD300(self.input_shape, num_classes=self.classes_nums)
        model.load_weights("./ckpt/weights_SSD300.hdf5", by_name=True)

        feature = []
        images_data = []
        # - 读取多个本地路径测试图片，preprocess_input以及保存图像像素值（显示需要）
        for path in os.listdir("./images"):
            img_path = os.path.join("./images/", path)
            # 1、输入到SSD网络当中，数组
            image = load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
            image = img_to_array(image)

            feature.append(image)
            # 2、读取图片二进制数据，matplotlib显示使用
            images_data.append(imread(img_path))

        # - 模型预测结果，得到7308个priorbox
        # 处理
        inputs = preprocess_input(np.asarray(feature))
        print(inputs)
        pred = model.predict(inputs)
        # (2, 7308, 33) 2代表图片数量， 7308代表每个图片预测的default boxes数量，33：4（位置） + 21（预测概率） + 8（其它default boxes参数）
        print(pred.shape)

        # - 进行非最大抑制算法处理NMS 21类别
        bb = BBoxUtility(self.classes_nums)
        res = bb.detection_out(pred)
        # (136, 6) (26, 6)
        print(res[0].shape, res[1].shape)
        # 200个候选框， 每个候选框位置，类别
        return res, images_data

    def tag_picture(self, images_data, outputs):
        """
        显示预测结果到图片中
        :return:
        """
        # 1、获取每张图片的预测结果中的值
        for i, img in enumerate(images_data):

            # 获取res当中对应的结果label, confidence:预测概率 location, xmin, ymin, xmax, ymax
            pre_label = outputs[i][:, 0]
            pre_conf = outputs[i][:, 1]
            pre_xmin = outputs[i][:, 2]
            pre_ymin = outputs[i][:, 3]
            pre_xmax = outputs[i][:, 4]
            pre_ymax = outputs[i][:, 5]

            # print("pre_label:{}, pre_conf:{}, pre_xmin:{}, pre_ymin:{}, pre_xmax:{}, pre_ymax:{}".
            #       format(pre_label, pre_conf, pre_xmin, pre_ymin, pre_xmax, pre_ymax))

            # 由于检测出的物体还是很多，所以进行显示过滤（90%）
            top_indices = [i for i, conf in enumerate(pre_conf) if conf > 0.6]
            top_conf = pre_conf[top_indices]
            top_label_indices = pre_label[top_indices].tolist()
            top_xmin = pre_xmin[top_indices]
            top_ymin = pre_ymin[top_indices]
            top_xmax = pre_xmax[top_indices]
            top_ymax = pre_ymax[top_indices]

            print("after filter top_label_indices:{}, top_conf:{}, top_xmin:{}, top_ymin:{}, top_xmax:{}, top_ymax:{}".
                  format(top_label_indices, top_conf, top_xmin, top_ymin, top_xmax, top_ymax))

            # matplotlib画图显示结果
            # 定义21中颜色，显示图片
            # currentAxis增加图中文本显示和标记显示
            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            plt.imshow(img / 255.)
            currentAxis = plt.gca()

            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * img.shape[1]))
                ymin = int(round(top_ymin[i] * img.shape[0]))
                xmax = int(round(top_xmax[i] * img.shape[1]))
                ymax = int(round(top_ymax[i] * img.shape[0]))

                # 获取该图片预测概率，名称，定义显示颜色
                score = top_conf[i]
                label = int(top_label_indices[i])
                label_name = self.classes_name[label - 1]
                display_txt = '{:0.2f}, {}'.format(score, label_name)
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                color = colors[label]
                # 显示方框
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # 左上角显示概率以及名称
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

            plt.show()


        return None


if __name__ == '__main__':
    ssd = SSDTest()

    outputs, images_data = ssd.test()

    # 显示图片
    ssd.tag_picture(images_data, outputs)