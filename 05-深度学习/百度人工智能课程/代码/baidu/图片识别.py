from aip import AipImageClassify
from pprint import pprint


def general_detect(client, image_path):
    """
    图片识别
    :param client: 实例化的aipImageClassify
    :param image_path: 图片路径
    :return: None
    """
    # 读取照片
    with open(image_path, 'rb') as f:
        img = f.read()
        # 调用检测函数
        result = client.advancedGeneral(img)
        pprint(result)

    return None

def detect_dish(client, image_path):
    """
    菜品识别
    :param client: AipImageClassify 实例
    :param image_path: 需要识别的菜品的图片路径
    :return: None
    """
    # 读取图片
    with open(image_path, 'rb') as f:
        image = f.read()

        option = {}
        option['top_num'] = 2
        # 调用菜品识别函数
        result = client.dishDetect(image, option)

        pprint(result)


    return None

def detect_car(client, image_path):
    """
    识别汽车品牌和型号
    :param client: AipImageClassify 实例
    :param image_path: 需要识别的汽车图片路径
    :return: None
    """

    # 打开图片， 读取图片
    with open(image_path, 'rb') as f:
        image = f.read()

        # 参数
        options = {}
        options['top_num'] = 2

        # 调用检测函数
        result = client.carDetect(image, options)

        pprint(result)

    return None

if __name__ == "__main__":
    # 实例化AipImageClassify
    aip = AipImageClassify('11536339', 'WdiZKiuizEOHjEuKHAdeEXBd', '6awIjKAyA7DhbCXWPvjIot8ry4udwryh')

    # 调用检测函数
    # general_detect(aip, '.\data\images\家居_05.jpg')

    # 调用菜品检测函数
    # detect_dish(aip, '.\data\images\菜品_035.jpg')

    # 调用汽车识别函数
    detect_car(aip, '.\data\images\宾利_05.jpg')

