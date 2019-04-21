import flask
from flask import Flask, request, render_template, url_for
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torchvision import transforms, models, datasets
from PIL import Image
import io as StringIO
import json
import os
from random import sample
from lib import Model, ClassAverages
from lib.DataUtils import Dataset
from lib.Utils import *
import cv2
import random
from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from random import randint

app = Flask(__name__)

@app.route("/")
def index(images_to_display=10):
    names = os.listdir(os.path.join('./static/Kitti', "validation", "image_2"))
    img_url_list = []
    rendering_images = sample(names, images_to_display)
    for i in rendering_images:
        img_url = "./static/Kitti/validation/image_2/{}".format(i)
        img_url_list.append(img_url)

    return flask.render_template('index.html', value = img_url_list)


@app.route('/predict', methods=['POST'])
def make_prediction():

    if request.method == 'POST':
        img_path = list(request.form.to_dict().keys())[0]
        img_key = img_path.split("/")[-1].split(".")[0]

         # Load Test Images from eval folder
        dataset = Dataset("./static/Kitti/validation")
        all_images = dataset.all_objects()
        print ("Length of eval data",len(all_images))
        averages = ClassAverages.ClassAverages()
        print ("Model is commencing predictions.....")


        all_images = dataset.all_objects()
        new_dict = {img_key: all_images[img_key]}
        print ("Length of eval data",len(new_dict))


        for key in new_dict.keys():
            data = all_images[key]
            truth_img = data['Image']
            img = np.copy(truth_img)
            imgGT = np.copy(truth_img)
            objects = data['Objects']
            cam_to_img = data['Calib']

            for object in objects:
                label = object.label
                theta_ray = object.theta_ray
                input_img = object.img

                input_tensor = torch.zeros([1,3,224,224])
                input_tensor[0,:,:,:] = input_img
                input_tensor.cuda()

                [orient, conf, dim] = model(input_tensor)
                orient = orient.cpu().data.numpy()[0, :, :]
                conf = conf.cpu().data.numpy()[0, :]
                dim = dim.cpu().data.numpy()[0, :]

                dim += averages.get_item(label['Class'])

                argmax = np.argmax(conf)
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += dataset.angle_bins[argmax]
                alpha -= np.pi

                location = plot_regressed_3d_bbox_2(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)
                locationGT = plot_regressed_3d_bbox_2(imgGT, truth_img, cam_to_img, label['Box_2D'], label['Dimensions'], label['Alpha'], theta_ray)


            file_ = randint(0, 25000)
            print(file_)
            cv2.imwrite("./static/temp/original-{}.png".format(str(file_)), truth_img, [cv2.IMWRITE_PNG_COMPRESSION, 10])
            cv2.imwrite("./static/temp/gt-{}.png".format(file_), imgGT, [cv2.IMWRITE_PNG_COMPRESSION, 10])
            cv2.imwrite("./static/temp/results-{}.png".format(file_), img, [cv2.IMWRITE_PNG_COMPRESSION, 10])
            img, truth_img, imgGT = None, None, None
            print("plotting done")

        return flask.render_template('visualize.html', file_num=file_)

def thresh_sort(x, thresh):
    idx, = np.where(x > thresh)
    return idx[np.argsort(x[idx])]

def init_model():
    print("Initializing model")
    np.random.seed(2019)
    torch.manual_seed(2019)
    my_vgg = models.vgg19_bn(pretrained=True)
    model = Model.Model(features=my_vgg.features, bins=2)
    checkpoint = torch.load("./weights/best_weights.pkl")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model created and weights loaded successfully")
    return model

if __name__ == '__main__':
    # initialize model
    model = init_model()

    # initialize labels
    # start app
    app.run(host='0.0.0.0', port=8000)
