# Cloud-Deployment-of-Sewer-Defect-Detection-Using-YOLOv8
Deploy the model on a cloud server to run the inference service, and perform image inference locally by calling the Python package designed for inference.

## 检测效果

![功能测试](https://github.com/allrivertosea/Cloud-Deployment-of-Sewer-Defect-Detection-Using-YOLOv8/blob/main/00005518_result.png)


## 使用说明

### 阿里云服务器


![图片](https://github.com/allrivertosea/Cloud-Deployment-of-Sewer-Defect-Detection-Using-YOLOv8/blob/main/pics/%E5%9B%BE%E7%89%87.png)
- 镜像为 Ubuntu 18.04 64 位，分配公网 IP。注意：在安全组中我们要添加入方向允许任何 IP 地址的流量访问某端口。
![图片](https://github.com/allrivertosea/Cloud-Deployment-of-Sewer-Defect-Detection-Using-YOLOv8/blob/main/pics/%E5%9B%BE%E7%89%871.png)
- 将推理服务代码上传到服务器，运行：
![图片](https://github.com/allrivertosea/Cloud-Deployment-of-Sewer-Defect-Detection-Using-YOLOv8/blob/main/pics/%E5%9B%BE%E7%89%872.png)

### 生成推理 Python 包

- 推理程序包：sewer_dt
Infer.py：我们定义一个 SewerDetector 类，构造方法中有两个实例属性（url 和 headers），有一个普通实例方法 detect，传输参数为 image_path，返回由云服务器得到的推理结果
result。
- 编写 setup.py
在封装代码 sewer_dt 同级目录下，编写一个 setup.py 脚本，用于描述自己的 Python 包。
- 编写 MANIFEST.in
- 打包 Python 包
在命令行中进入封装代码的根目录，并执行以下命令来打包 Python 包：
python setup.py sdist bdist_wheel，将在 dist 目录下生成一个.tar.gz 文件和一个.whl 文件，用于安装和发布 Python 包。
- 安装 Python 包
可以将其发布到 PyPI 上，这里我们只进行本地安装。
pip install dist/sewer_dt-0.1.tar.gz
### 本地使用 sewer_dt 包进行推理
python infer_from_package.py


