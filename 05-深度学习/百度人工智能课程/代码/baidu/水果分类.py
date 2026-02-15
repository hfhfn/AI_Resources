# 获取access token
# 调用api 进行分类

import urllib.request
import json
from pprint import pprint
import base64


# 定义Fruit 类
class Fruit(object):
    def __init__(self, app_id, api_key, secret_key):
        self.app_id = app_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.token_url = 'https://aip.baidubce.com/oauth/2.0/token'
        self.service_url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/itcast_fruit'

    def get_token(self):
        """
        获取 access token
        :return: access token
        """
        grant_type = 'client_credentials'
        client_id = self.api_key
        client_secret = self.secret_key

        # 构建url地址和参数
        url = self.token_url + '?grant_type=' + grant_type + '&client_id=' + client_id + '&client_secret=' + client_secret

        # 构建http 请求
        request = urllib.request.Request(url)
        request.add_header('Content-Type', 'application/json; charset=UTF-8')

        # 发送http请求
        response = urllib.request.urlopen(request)
        content = response.read()

        # pprint(json.loads(content.decode()))
        return json.loads(content.decode())['access_token']

    def classify(self, token, image):
        """
        对水果图片进行分类
        :param token: access token
        :param image: 已经读取好的图片数据, 还未进行base64编码
        :return: 返回分类结果
        """
        # 构建服务url + 参数: access_token
        url = self.service_url + '?access_token=' + token

        # 对图片进行base64编码
        img64 = base64.b64encode(image).decode()

        # 构建 http 请求 body
        params = {}
        params['image'] = img64
        params['top_num'] = 2

        # 把字典数据编码为字符串
        params = json.dumps(params).encode()

        # 构建http 请求
        request = urllib.request.Request(url, params)
        request.add_header('Content-Type', 'application/json; charset=UTF-8')

        # 发送请求
        response = urllib.request.urlopen(request)
        content = response.read()
        result = json.loads(content.decode())
        return result


if __name__ == "__main__":
    # 实例化Fruit类
    fruit = Fruit('11536339', 'WdiZKiuizEOHjEuKHAdeEXBd', '6awIjKAyA7DhbCXWPvjIot8ry4udwryh')

    # 调用get_token获取access_token
    access_token = fruit.get_token()
    # print(access_token)

    # 读取图片, 调用classify函数
    with open('./data/images/banana1.jpg', 'rb') as f:
        image = f.read()
        ret = fruit.classify(access_token, image)
        print(ret)