import PIL.Image
import torch
from torchvision.transforms.functional import to_pil_image
from TransferModel.AdaAttN.adaattn_model import AdaAttNModel
from TransferModel.AdaIN.adain_model import AdaINModel
from hyper import *
from torchvision import transforms
import io
import base64
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL

def img2str(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def tensor_to_pil(image_tensor):
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.clamp(image_tensor, 0, 1) 
        return to_pil_image(image_tensor)
    
    elif isinstance(image_tensor, tf.Tensor):
        image_tensor = np.array(image_tensor*255, dtype=np.uint8)
        if np.ndim(image_tensor)>3:
            assert image_tensor.shape[0] == 1
            image_tensor = image_tensor[0]
        return PIL.Image.fromarray(image_tensor)

def resize_to_even(image, target_size):
    width, height = image.size

    if width < height:
        new_width = target_size
        new_height = int((height / width) * new_width)
    else:
        new_height = target_size
        new_width = int((width / height) * new_height)

    if new_height % 32 != 0:
        new_height = (new_height // 32) * 32 

    if new_width % 32 != 0:
        new_width = (new_width // 32) * 32 

    return image.resize((new_width, new_height))


def adaattn_preprocess(img):
    resized_img = resize_to_even(img, target_size=IMG_SIZE)
    transformed_img = transforms.ToTensor()(resized_img)
    return transformed_img.unsqueeze(0)

def adain_preprocess(img):
    resized_img = transforms.Resize(IMG_SIZE)(img)
    transformed_img = transforms.ToTensor()(resized_img)
    return transformed_img.unsqueeze(0)

def tf_preprocess(image):
    img = np.array(image)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img / 255.0 

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = IMG_SIZE / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def load_model():
    return {
        AdaAttN : {
            'model' : AdaAttNModel(encoder_path=AdaAttN_encoder, decoder_path=AdaAttN_decoder, adaattn_3_path=AdaAttN_adattn_3, adaattn_4_5_path=AdaAttN_adattn_4_5),
            'preprocess': adaattn_preprocess
            },
        AdaIN : {
            'model' : AdaINModel(encoder_path=AdaIN_encoder, decoder_path=AdaIN_decoder),
            'preprocess' : adain_preprocess
            },
        TFStyleTransfer : {
            'model' : hub.load(TF_model),
            'preprocess' : tf_preprocess
        }
    }