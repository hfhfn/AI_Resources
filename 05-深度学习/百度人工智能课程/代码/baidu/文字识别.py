from aip import AipOcr
from pprint import pprint

def general_ocr(client, image_path):
    """
    通用文字识别函数
    :param client: AipOcr实例
    :param image_path: 图片路径
    :return: None
    """
    # 以二进制格式打开图片文件
    with open(image_path, 'rb') as f:
        # 读取图片
        image = f.read()
        # 设定参数
        params = {}
        params['detect_direction'] = 'true' # True
        params['probability'] = 'true'

        # 调用识别函数
        result = client.basicGeneral(image, params)
        pprint(result)

    return None

def plate_ocr(client, image_path):
    """
    车牌识别
    :param client: AipOcr对象
    :param image_path: 图片路径
    :return: None
    """
    with open(image_path, 'rb') as f:
        options = {}
        options['multi_detect'] = 'false'

        result = client.licensePlate(f.read(), options)
        pprint(result)

def receipt_ocr(client, image_path):
    """
    通用票据识别
    :param client: AipOcr 实例
    :param image_path: 图片路径
    :return: None
    """
    with open(image_path, 'rb') as f:
        params = {}
        params['recognize_granularity'] = 'big'
        params['probability'] = 'true'
        params['accuracy'] = 'normal'

        result = client.receipt(f.read(), params)
        pprint(result)

    return None

def custom_ocr(client, image_path):
    """
    调用自定义模板识别票据
    :param client: AipOcr 实例
    :param image_path: 图片路径
    :return: None
    """
    with open(image_path, 'rb') as f:
        tempSign = '0a8ff37f5c6e500ab3356141115db72d'

        result = client.custom(f.read(), tempSign)
        pprint(result)

    return None

if __name__ == "__main__":
    # 实例化AipOcr
    aip = AipOcr('11560811', 'gN0pPnWTQsuvDSrGXptq5Zyl', 'dVaefK5FYkfD1AxhYOb9dg2EeDSBLKEj')

    # 调用自定义的识别函数
    # general_ocr(aip, './data/images/报纸2.png')

    # plate_ocr(aip, './data/images/plate.jpg')

    # receipt_ocr(aip, './data/images/增值税发票.jpg')
    custom_ocr(aip, './data/images/增值税发票_江西.jpg')