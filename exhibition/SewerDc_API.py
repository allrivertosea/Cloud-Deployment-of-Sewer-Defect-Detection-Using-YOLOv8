# -*- coding: utf-8 -*-

from flask import Flask, request, make_response
from detection_model import YOLO
import json
import argparse
import os
from io import BytesIO
from PIL import Image
app = Flask(__name__)


parse = argparse.ArgumentParser(description='onnx model infer!')
parse.add_argument('--weights_path', type=str, default=r'weights/best.onnx', help='onnx模型存放路径')
parse.add_argument('--savepath', type=str, default=r'runs/detect/pics_results', help='存储路径，视频为: video_results')
args = parse.parse_args()
class_name = ['CR', 'ZW', 'JB', 'JS', 'PL']

# 加载模型
model = YOLO(model=args.weights_path)

# 处理请求
@app.route('/yolov8', methods=['POST'])
def hello():
    img = request.data
    img_bytes = BytesIO(img)
    one_image_data = Image.open(img_bytes)
    results = model.predict(source=one_image_data, save=False, save_path=args.savepath, device='cpu', conf=0.8,show=False, verbose=False)

    try:
        prob = round(results[0].boxes.conf.tolist()[0], 4)
        class_index = int(results[0].boxes.cls.tolist()[0])
        one_image_result =  '缺陷类别为：' + class_name[class_index] + ',' + '概率为：' + str(prob)
    except Exception as e:
        one_image_result ='正常图片'
    rsp = make_response(json.dumps(one_image_result,ensure_ascii=False))
    rsp.headers['Content-Type'] = 'application/json; charset=utf-8'
    return rsp


if __name__ == '__main__':

    # app.debug = True
    app.run(host='0.0.0.0', port='80')
