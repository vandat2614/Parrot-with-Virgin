document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btn_download').addEventListener('click', () => {

        const outputImage = document.getElementById('output_image');
        const imageUrl = outputImage.src; 
        const link = document.createElement('a');
        link.href = imageUrl;

        link.download = 'output_image.jpg';
        link.click();
    });
});
