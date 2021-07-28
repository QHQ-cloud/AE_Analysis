# -*- coding: utf-8 -*-
# 导入常用的库
import time
import os
import cv2
import numpy as np
# 导入flask库
from torchvision import transforms as T
import torch
import flask
from flask import Flask, render_template, request, jsonify
from C3D_denseNeXt_withSEModule import DenseNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 导入pickle库
import pickle
from torch.nn import functional as F
with open('./ae_class.txt', 'r') as f:
    idx2label = eval(f.read())
# 实例化Flask对象
app = Flask(__name__)
# 设置开启web服务后，如果更新html文件，可以使更新立即生效
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model  # 下面的那个predict也是要用的 所以在这里定义为全局
    model = DenseNet(n_input_channels=1, num_init_features=64,
                     growth_rate=32,
                     block_config=(3, 6, 12, 8), num_classes=4).to(device)
    model.load_state_dict(torch.load("./model29.pkl"))
    model.eval()

 
# 定义函数classId_to_className，把种类索引转换为种类名称
def classId_to_className(classId):
    category_list = ['green', 'yellow', 'orange', 'red']
    className = category_list[classId]
    return className
 
def prepare_image(buffer):
    buffer = buffer[:, :, :, :1]
    for i, frame in enumerate(buffer):  # [None,1000,500,1]
        frame -= np.array([[[248.1185]]])
        frame /= np.array([[[37.7234]]])
        frame = cv2.GaussianBlur(frame, (7, 7), 1)
        frame = np.expand_dims(frame, axis=-1)
        buffer[i] = frame
    buffer_resize = np.empty((16, 448, 224, 1), np.dtype('float32'))
    for i, frame in enumerate(buffer):  # [1000,500,1]
        frame = cv2.resize(frame, (224, 448))  # numpy.ndarray [224,224,3]
        frame = np.expand_dims(frame, axis=-1)
        buffer_resize[i] = frame
    buffer_resize = buffer_resize.transpose((3, 0, 1, 2))

    buffer_resize = torch.from_numpy(buffer_resize)
    buffer_resize = buffer_resize.to(device)
    buffer_resize = buffer_resize.unsqueeze(dim = 0)
    print(buffer_resize.shape)
    return buffer_resize

# 开启服务

    # Initialize the data dictionary that will be returned from the view.


@app.route('/')
def index_page():
    return render_template('_05_web_page.html')

@app.route("/predict_image", methods=['POST'])
def any_name_you_like():

    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.form.get('input_image'):
            # Read the image in PIL format
            img_path = request.form.get('input_image')
            print(img_path)
            # print(image_dir)
            frames = sorted([os.path.join(img_path, img) for img in os.listdir(img_path)],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
            frame_count = len(frames)  # 16
            # 这是之前的图片需要进行crop 但是目前不需要
            buffer = np.empty((frame_count, 1000, 500, 3), np.dtype('float32'))
            # 1118 和 673 是 原始图片的大小
            for i, frame_name in enumerate(frames):
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)  # [1000, 500, 3]
                buffer[i] = frame

            images = prepare_image(buffer)

            preds = F.softmax(model(images), dim=1)  # [1,10]

            results = torch.topk(preds.cpu().data, k=3, dim=1)  # test for it
            results = (results[0].cpu().numpy(), results[1].cpu().numpy())
            # 假设这里针对一张图片的输入 得到的results 底下跟topk有关系
            data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            for prob, label in zip(results[0][0], results[1][0]):
                label_name = idx2label[label]
                r = {"label": label_name, "probability": float(round(prob,2))}
                data['predictions'].append(r)
            data["success"] = True
    # D:\modelling\dataset_classify\train\label_2\AEPic_1354_8x_139
    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


# 根据图片文件的路径获取图像矩阵

# 主函数        
if __name__ == "__main__":
    load_model()
    app.run("127.0.0.1", port=5000)
    