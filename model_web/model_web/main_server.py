import os
import json
# 安装所需工具包
import flask
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torch.autograd import Variable
from C3D_denseNeXt_withSEModule import DenseNet
# 初始化Flask app
app = flask.Flask(__name__)
model = None
use_gpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 返回结果用的
with open('./ae_class.txt', 'r') as f:
    idx2label = eval(f.read())
# 实际上的id与真实对应的一个字典
# 加载模型进来
def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model  # 下面的那个predict也是要用的 所以在这里定义为全局
    model = DenseNet(n_input_channels=1, num_init_features=64,
                     growth_rate=32,
                     block_config=(3, 6, 12, 8), num_classes=4).to(device)
    model.load_state_dict(torch.load("./model29.pkl"))
    model.eval()

# 数据预处理
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
@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.form.get("data"):
            # Read the image in PIL format
            image_dir = flask.request.form.get("data",type = str)
            # print(image_dir)
            frames = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
            frame_count = len(frames)  # 16
            # 这是之前的图片需要进行crop 但是目前不需要
            buffer = np.empty((frame_count, 1000, 500, 3), np.dtype('float32'))
            # 1118 和 673 是 原始图片的大小
            for i, frame_name in enumerate(frames):
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)  # [1000, 500, 3]
                buffer[i] = frame
            # image = Image.open(io.BytesIO(image)) #二进制数据

            # Preprocess the image and prepare it for classification.
            images = prepare_image(buffer)

            # Classify the input image and then initialize the list of predictions to return to the client.
            preds = F.softmax(model(images), dim=1)  # [1,10]
            # results = torch.max(preds,dim=1)# test for it
            #
            # label_name = idx2label[results[1].cpu().numpy()[0]]
            results = torch.topk(preds.cpu().data, k=3, dim=1)  # test for it
            results = (results[0].cpu().numpy(), results[1].cpu().numpy())
            # 假设这里针对一张图片的输入 得到的results 底下跟topk有关系
            data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            for prob, label in zip(results[0][0], results[1][0]):
                label_name = idx2label[label]
                r = {"label": label_name, "probability": float(prob)}
                data['predictions'].append(r)
            data["success"] = True
    # D:\modelling\dataset_classify\train\label_2\AEPic_1354_8x_139
    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run()
