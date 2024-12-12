# Cloud-Deployment-of-Sewer-Defect-Detection-Using-YOLOv8
Deploy the model on a cloud server to run the inference service, and perform image inference locally by calling the Python package designed for inference.

## 检测效果

![功能测试](https://github.com/allrivertosea/Cloud-Deployment-of-Sewer-Defect-Detection-Using-YOLOv8/blob/main/00005518_result.png)


## 环境配置

conda create -n infer_env python=3.8 -y
conda activate infer_env
pip install -r requirements.txt

## 使用说明

```
python xxx.py   #执行推理操作
```


