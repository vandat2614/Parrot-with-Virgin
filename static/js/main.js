function startTypingEffect(elementId, text, typingSpeed = 100, resetInterval = 10000) {
    const typingEffect = document.getElementById(elementId);
    let index = 0;
  
    function type() {
      if (index < text.length) {
        typingEffect.textContent += text[index];
        index++;
        setTimeout(type, typingSpeed);
      }
    }
  
    function resetTypingEffect() {
      typingEffect.textContent = "";
      index = 0;
      type();
    }
  
    type();
    setInterval(resetTypingEffect, resetInterval);
  }
  
  startTypingEffect("typingEffect", "Let's get started!", 100, 10000);
  



function displayImage(file, containerId) {
    const imageBox = document.getElementById(containerId);
    const imageContainer = imageBox.querySelector('.image-container');
    imageContainer.innerHTML = '';

    const reader = new FileReader();
    reader.onload = function(event) {
        const img = document.createElement('img');
        img.src = event.target.result;
        imageContainer.appendChild(img);
    };
    reader.readAsDataURL(file);
}

// Function to handle file drop and update input
function handleFileDrop(event, containerId, inputId) {
    event.preventDefault();
    event.target.classList.remove('dragging');
    const file = event.dataTransfer.files[0];
    if (file) {
        displayImage(file, containerId);

        // Assign file to input element
        const inputElement = document.getElementById(inputId);
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        inputElement.files = dataTransfer.files;
    }
}

// Handle file selection for upload
document.getElementById('style').addEventListener('change', function(event) {
    displayImage(event.target.files[0], 'style-dropzone');
});

document.getElementById('content').addEventListener('change', function(event) {
    displayImage(event.target.files[0], 'content-dropzone');
});

// Handle drag and drop functionality for style-dropzone
const styleDropzone = document.getElementById('style-dropzone');
styleDropzone.addEventListener('dragover', function(event) {
    event.preventDefault();
    styleDropzone.classList.add('dragging');
});
styleDropzone.addEventListener('dragleave', function(event) {
    styleDropzone.classList.remove('dragging');
});
styleDropzone.addEventListener('drop', function(event) {
    handleFileDrop(event, 'style-dropzone', 'style');
});

// Handle drag and drop functionality for content-dropzone
const contentDropzone = document.getElementById('content-dropzone');
contentDropzone.addEventListener('dragover', function(event) {
    event.preventDefault();
    contentDropzone.classList.add('dragging');
});
contentDropzone.addEventListener('dragleave', function(event) {
    contentDropzone.classList.remove('dragging');
});
contentDropzone.addEventListener('drop', function(event) {
    handleFileDrop(event, 'content-dropzone', 'content');
});


/*Hide = Show*/
document.addEventListener('DOMContentLoaded', async () => {
    // Khai báo các container và nút
    const galleryContainers = {
        content: document.querySelector('.started_2_flex'),
        style: document.querySelector('.started_3_flex')
    };

    const seeMoreBtns = {
        content: document.querySelector('#see-more-btn1'),
        style: document.querySelector('#see-more-btn2')
    };

    const hideBtns = {
        content: document.querySelector('#hide-btn1'),
        style: document.querySelector('#hide-btn2')
    };

    const imageContainers = {
        content: document.querySelector('#content-dropzone .image-container'),
        style: document.querySelector('#style-dropzone .image-container')
    };

    const inputElements = {
        content: document.getElementById('content'),
        style: document.getElementById('style')
    };

    // Hàm tải ảnh từ API và hiển thị chúng
    async function loadImages(apiUrl, container, type) {
        const response = await fetch(apiUrl);
        const images = await response.json();

        const firstImages = images.slice(0, 3);
        firstImages.forEach(img => {
            const imgElement = createImageElement(img, false, type);
            container.appendChild(imgElement);
        });

        const remainingImages = images.slice(3);
        remainingImages.forEach(img => {
            const imgElement = createImageElement(img, true, type);
            container.appendChild(imgElement);
        });
    }

    // Hàm tạo phần tử ảnh và thêm sự kiện click
    function createImageElement(src, hidden = false, type) {
        const imgElement = document.createElement('img');
        imgElement.src = src;
        imgElement.alt = src;
        imgElement.classList.add('img_sample');
        if (hidden) imgElement.classList.add('hidden');
        
        // Thêm sự kiện khi click vào ảnh
        imgElement.addEventListener('click', () => {
            displayImageInContainer(src, type);
        });

        return imgElement;
    }

    // Hàm hiển thị ảnh trong container và lưu vào thẻ input khi click vào ảnh
    async function displayImageInContainer(src, type) {
        const container = imageContainers[type];
        container.innerHTML = ''; // Xóa ảnh cũ nếu có
        const imgElement = document.createElement('img');
        imgElement.src = src;
        imgElement.alt = src;
        container.appendChild(imgElement);

        // Fetch ảnh từ URL, chuyển đổi thành file và lưu vào input
        const response = await fetch(src);
        const blob = await response.blob();
        const file = new File([blob], `selected_${type}.jpg`, { type: blob.type });
        
        // Cập nhật vào thẻ input để gửi form
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        inputElements[type].files = dataTransfer.files;
    }

    // Hàm xử lý sự kiện khi bấm "See More" hoặc "Hide"
    function toggleImages(seeMoreBtn, hideBtn, container) {
        seeMoreBtn.addEventListener('click', (event) => {
            event.preventDefault();
            container.querySelectorAll('.img_sample.hidden').forEach(img => img.classList.remove('hidden'));
            seeMoreBtn.style.display = 'none';
            hideBtn.classList.remove('hidden');
        });

        hideBtn.addEventListener('click', (event) => {
            event.preventDefault();
            container.querySelectorAll('.img_sample').forEach((img, index) => {
                if (index >= 3) img.classList.add('hidden');
            });
            hideBtn.classList.add('hidden');
            seeMoreBtn.style.display = 'inline';
        });
    }

    // Load ảnh cho từng container và thiết lập sự kiện
    await loadImages('/api/content_images', galleryContainers.content, 'content');
    await loadImages('/api/style_images', galleryContainers.style, 'style');

    toggleImages(seeMoreBtns.content, hideBtns.content, galleryContainers.content);
    toggleImages(seeMoreBtns.style, hideBtns.style, galleryContainers.style);
});


document.getElementById('transfer-btn').addEventListener('click', function(event) {

    document.getElementById('overlay').style.display = 'flex';
    setTimeout(function() {
        document.getElementById('overlay').style.opacity = 1;  // Làm mờ màn hình
    }, 10);
});