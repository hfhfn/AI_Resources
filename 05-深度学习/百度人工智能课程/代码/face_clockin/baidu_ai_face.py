import time
import urllib.request
import json
import base64
from pprint import pprint

class BaiduFace(object):
    def __init__(self, app_id, api_key, secret_key):
        self.app_id = app_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.token = None
        self.time_exp = None # 超时时间
        self.error = 0
        self.err_msg = ''

    def _get_token(self):
        """
        获取token
        :return: token
        """
        self.error = 0
        self.err_msg = ''

        # 判断是否有有效token
        if self.token == None or time.time() > self.time_exp:
            # url
            url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' \
                  + self.api_key + '&client_secret=' + self.secret_key

            # 构建 url 请求
            request = urllib.request.Request(url)
            request.add_header('Content-Type', 'application/json;charset=UTF-8')

            # 发送请求
            response = urllib.request.urlopen(request)

            # 解析返回， 去除token
            content = response.read()
            content_decoded = json.loads(content.decode())

            if 'access_token' not in content_decoded.keys():
                self.error = 1
                self.err_msg = content_decoded['error']

            self.token = content_decoded['access_token']
            self.time_exp = time.time() + content_decoded['expires_in'] // 2

        return self.error


    def add_user(self, img64, groupid, userid, userinfo=None):
        """
        添加人脸图片到百度人脸库中
        :param img64: 经过base64编码的照片
        :param groupid: 用户组id
        :param userid: 用户id
        :param userinfo: 用户信息
        :return: 成功返回0, 失败为错误码
        """
        self.error = 0
        self.err_msg = ''

        # 获取token
        if self._get_token() != 0:
            print(self.err_msg)
            return self.error

        # 填写参数
        params = {}
        params['image'] = img64
        params['image_type'] = 'BASE64'
        params['group_id'] = groupid
        params['user_id'] = userid
        if userinfo:
            params['user_info'] = userinfo
        params['quality_control'] = 'NORMAL'
        params['liveness_control'] = 'LOW'

        params = json.dumps(params).encode()

        # 发送http请求
        url = 'https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add' + '?access_token=' + self.token
        request = urllib.request.Request(url, params)
        request.add_header('Content-Type', 'application/json;charset=UTF-8')

        response = urllib.request.urlopen(request)

        # 分析结果
        content = response.read()
        content_decoded = json.loads(content.decode())

        if 'error_code' in content_decoded.keys():
            self.error = content_decoded['error_code']
            self.err_msg = content_decoded['error_msg']

        return self.error


    def search_user(self, img64, groupid_list):
        """
        人脸搜索函数
        :param img64: 人脸照片
        :param groupid_list: 组列表
        :return: 成功返回用户信息， 失败返回错误代码和信息
        """
        # 获取token
        if self._get_token() != 0:
            print(self.err_msg)
            return self.error

        # 填写参数
        params = {}
        params['image'] = img64
        params['image_type'] = 'BASE64'
        params['group_id_list'] = groupid_list
        params['quality_control'] = 'NORMAL'
        params['liveness_control'] = 'LOW'

        params = json.dumps(params).encode()

        # 构建和发送请求
        url = 'https://aip.baidubce.com/rest/2.0/face/v3/search' +'?access_token=' + self.token
        request = urllib.request.Request(url, params)
        request.add_header('Content-Type', 'application/json; charset=UTF-8')

        response = urllib.request.urlopen(request)

        # 分析返回结果
        con = json.loads(response.read().decode())

        pprint(con)

        if con['error_code'] != 0:
            return {'error':con['error_code'], 'err_msg':con['error_msg']}

        user = con['result']['user_list'][0]
        user['error'] = 0

        return user

bf = BaiduFace('11511733', 'geCvIhXiRUm3KqzGozC9lPms', 'dRPukMIbLAfSos7yqzOP4Linpb25g9V0')

if __name__ == '__main__':
    # ret = bf._get_token()

    # print(bf.token)
    with open('02.png', 'rb') as f:
        img = f.read()

        img64 = base64.b64encode(img).decode()

        # ret = bf.add_user(img64, 'stars', 'liuyifei')
        ret = bf.search_user(img64, 'stars')
        print(ret)
