import requests

class SewerDetector:
    def __init__(self):
        self.url = 'http://8.134.147.207:80/yolov8'
        self.headers = {'Content-Type': 'image/png'}

    def detect(self, image_path):
        # 读取图片文件
        with open(image_path, 'rb') as f:
            image_data = f.read()
        # 发送POST请求
        response = requests.post(self.url, data=image_data, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        result = response.text
        return result


if __name__ == '__main__':
    image_path = r'D:\exhibition\datasets\infer_data\00000155.png'
    sd = SewerDetector()
    result = sd.detect(image_path)
    print(result)
