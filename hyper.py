IMG_SIZE = 512
CONTENT_IMAGE_FOLDER = 'static\images/content'
STYLE_IMAGE_FOLDER = 'static\images\style'

AdaAttN = 'AdaAttN'
AdaAttN_encoder = 'TransferModel/AdaAttN/vgg_normalised.pth'
AdaAttN_decoder = 'TransferModel\AdaAttN\latest_net_decoder.pth'
AdaAttN_adattn_3 = 'TransferModel\AdaAttN\latest_net_adaattn_3.pth'
AdaAttN_adattn_4_5 = 'TransferModel\AdaAttN\latest_net_transformer.pth'

AdaIN = 'AdaIN'
AdaIN_encoder = 'TransferModel\AdaIN/vgg_normalised.pth'
AdaIN_decoder = 'TransferModel\AdaIN/adain_decoder.pth'

TFStyleTransfer = 'TF-StyleTransfer'
TF_model = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'