import os
IMG_SIZE = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'content')
STYLE_IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'style')

AdaAttN = 'AdaAttN'
AdaAttN_encoder = 'TransferModel/AdaAttN/vgg_normalised.pth'
AdaAttN_decoder = 'TransferModel/AdaAttN/latest_net_decoder.pth'
AdaAttN_adattn_3 = 'TransferModel/AdaAttN/latest_net_adaattn_3.pth'
AdaAttN_adattn_4_5 = 'TransferModel/AdaAttN/latest_net_transformer.pth'

AdaIN = 'AdaIN'
AdaIN_encoder = AdaAttN_encoder
AdaIN_decoder = 'TransferModel/AdaIN/adain_decoder.pth'

# TFStyleTransfer = 'TF-StyleTransfer'
# TF_model = 'https://kaggle.com/models/google/arbitrary-image-stylization-v1/TensorFlow1/256/1'
