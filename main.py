import mxnet,warnings
import os,math,gluoncv
import streamlit as st
from decord import VideoReader
from gluoncv.data.transforms import video
from mxnet import nd,gluon
from mxnet.gluon import nn
warnings.filterwarnings("ignore")

maps={}
characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(len(characters)):
    maps[i]=characters[i]

def prediction(location):
    frame_resize = gluon.data.vision.transforms.Resize(256)  # Resizing each frame in the video to a fixed size.
    frame_normalize = video.VideoToTensor()  # Normalizing the values to be between 0 and 1.
    frame_transforms = gluon.data.vision.transforms.Compose(
        [frame_resize, frame_normalize])  # Composing the two transforms into a single one.

    frames = VideoReader(location)  # Reading the videos.
    length = len(frames)  # Getting the number of frames in the video.
    if length<28:
        d=1
        end=length
    else:
        d = math.floor(((length - 1) / (
                28 - 1)))  # Using AP (Arithmetic Progression), calculating the common difference between the frames.
        end = 1 + (28 - 1) * d  # Calculating the end value.
    frame_id_list = range(0, end, d)  # Getting the frames specified by the common difference.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gluon.nn.SymbolBlock.imports("model-symbol.json", ['data'], "model-0000.params", ctx=mxnet.cpu(0))
    prediction = (model((nd.expand_dims(nd.stack(*frame_transforms(nd.array(frames.get_batch(frame_id_list).asnumpy()))),
                           axis=0).transpose((0, 2, 1, 3, 4))).as_in_context(mxnet.cpu(0))).argmax(axis=-1)).asscalar()

    return maps[prediction]

st.image('isl.jpg')
video_file = st.file_uploader("Upload your video")
if video_file:
    with open(os.path.join("tempDir", video_file.name), "wb") as f:
        f.write(video_file.getbuffer())
    pred = st.button("Predict")
    if pred:
        st.text("The signed character is "+ prediction(os.path.join("tempDir", video_file.name)))