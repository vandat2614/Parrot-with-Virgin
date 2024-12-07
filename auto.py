import os
import argparse
from utils import load_model, tensor_to_pil
from PIL import Image

def load_image(image_path):
    """Load an image."""
    image = Image.open(image_path).convert('RGB')
    return image

def save_image(image, output_path):
    """Save the processed image."""
    image.save(output_path)

def process_images(content_dir, style_dir, method, size):
    """Process all pairs of content and style images using the specified method."""
    models = load_model()
    
    if method not in models:
        raise ValueError(f"Method '{method}' not found in available models.")

    preprocess_func = models[method]['preprocess']
    model = models[method]['model']

    output_dir = f"{os.path.basename(content_dir)}_{os.path.basename(style_dir)}_{method}"
    os.makedirs(output_dir, exist_ok=True)

    content_images = sorted(os.listdir(content_dir))
    style_images = sorted(os.listdir(style_dir))

    for content_image_name in content_images:
        content_image_path = os.path.join(content_dir, content_image_name)
        content_image = load_image(content_image_path)

        for style_image_name in style_images:
            style_image_path = os.path.join(style_dir, style_image_name)
            style_image = load_image(style_image_path)

            # Preprocess images
            processed_content = preprocess_func(content_image, size)
            processed_style = preprocess_func(style_image, size)

            # Apply style transfer
            output_image = model(processed_content, processed_style)
            output_image = tensor_to_pil(output_image[0])

            # Save the output image
            output_name = f"{os.path.splitext(content_image_name)[0]}_{os.path.splitext(style_image_name)[0]}.png"
            output_path = os.path.join(output_dir, output_name)
            save_image(output_image, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer Script")
    parser.add_argument("--content_dir", type=str, required=True, help="Folder containing content images.")
    parser.add_argument("--style_dir", type=str, required=True, help="Folder containing style images.")
    parser.add_argument("--method", type=str, default="AdaIN", help="Style transfer method to use. Default is 'default_method'.")
    parser.add_argument("--size", type=int, default=512, help="Resize images to this size before processing. Default is 256.")


    args = parser.parse_args()

    process_images(args.content_dir, args.style_dir, args.method, args.size)
