from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
from utils import *
import argparse
import lpips
import pandas as pd

# content_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/content"
# result_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/result"

content_dir = ""
result_dir = ""
metric = []

def preprocess_img(img):
    transformed_img = transforms.ToTensor()(img)
    return transformed_img.unsqueeze(0)

def calculate_ssim(image1, image2):
    # Chuyển đổi PIL Image sang numpy array
    img1_array = np.array(image1)
    img2_array = np.array(image2)
    
    # Tính SSIM cho ảnh nhiều kênh
    ssim_value = ssim(img1_array, img2_array, multichannel=True, data_range=255, channel_axis=-1)
    
    return ssim_value

def get_content_image(result_image_name):
    content_image_name = result_image_name.split('+')[0]
    for image_name in os.listdir(content_dir):
        if image_name.split(".")[0] == content_image_name: 
            content_image_name = image_name
            break
    return Image.open(os.path.join(content_dir, content_image_name)).convert("RGB")

def create_table(metric_avg):
    metric_avg_df = pd.DataFrame(metric_avg)
    return metric_avg_df.set_index('model')[[i+"_avg" for i in metric]]
   


def evaluate():
    loss_fn = lpips.LPIPS(net='vgg')
    metric_avg = []
    ssim_flag = ("ssim" in metric)
    lpips_flag = ("lpips" in metric)

    for model_name in os.listdir(result_dir):
        ssim = 0
        lpips_score = 0
        i = 0
        model_name_path = os.path.join(result_dir, model_name)
        print("-"*80,"\n- We are in", model_name_path)

        for category_style in os.listdir(model_name_path):
            category_style_path = os.path.join(model_name_path, category_style)

            for result_image_name in os.listdir(category_style_path):
                print(i)
                result_image = Image.open(os.path.join(category_style_path, result_image_name)).convert("RGB")
                content_image = resize_to_even(get_content_image(result_image_name), IMG_SIZE) if model_name == "AdaAttN" else transforms.Resize((result_image.size[1], result_image.size[0]))(get_content_image(result_image_name))
                if ssim_flag:
                    ssim += calculate_ssim(content_image, result_image)
                if lpips_flag:
                    lpips_score += loss_fn(preprocess_img(content_image), preprocess_img(result_image)).item()
                i += 1
                if i % 100 == 0: print ("+", i, "pair of image has been calculated!")
            # Kiem tra ham co chay duoc khong
            #     if i == 10: 
            #         break
            # if i == 10: break
        metric_avg.append(  {"model" : model_name, 
                            "ssim_avg" : float(ssim / i) if ssim_flag else None, 
                            "lpips_avg" : float(lpips_score / i) if lpips_flag else None,
                            "number_image" : i})

    return metric_avg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, required=True, help="Folder containing content images.")
    parser.add_argument("--result_dir", type=str, required=True, help="Folder containing result images.")
    parser.add_argument("--metric", type=lambda s: s.split(','), required=True, help="Evaluate metric")

    args = parser.parse_args()

    content_dir = args.content_dir
    result_dir = args.result_dir
    metric = args.metric
    metric_avg = evaluate()
    print(metric_avg)
    print(create_table(metric_avg))

# python evaluate.py  --content_dir D:/CS406-ImageProcessingAndApplication/Parrots/images/content --result_dir D:/CS406-ImageProcessingAndApplication/Parrots/images/result --metric lpips,ssim