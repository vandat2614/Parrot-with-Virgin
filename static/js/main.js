document.addEventListener('DOMContentLoaded', () => {
    const text = "Let's get started!";
    const typedTextElement = document.getElementById('typed-text');
    const cursorElement = document.querySelector('.cursor');
    let index = 0;
  
    function typeWriter() {
        if (index < text.length) {
            typedTextElement.textContent += text[index]; 
            index++;
            setTimeout(typeWriter, 150);
        } else {
            cursorElement.style.display = 'none';
        }
    }
  
    typeWriter();
});

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
    const galleryContainer1 = document.querySelector('.started_2_flex');
    const galleryContainer2 = document.querySelector('.started_3_flex');

    const seeMoreBtn1 = document.querySelector('#see-more-btn1'); 
    const seeMoreBtn2 = document.querySelector('#see-more-btn2'); 
    const hideBtn1 = document.querySelector('#hide-btn1'); 
    const hideBtn2 = document.querySelector('#hide-btn2'); 

    // Lấy danh sách hình ảnh từ API cho thư mục content
    const response_content = await fetch('/api/content_images');
    const images_content = await response_content.json();

    // Lấy danh sách hình ảnh từ API cho thư mục style
    const response_style = await fetch('/api/style_images');
    const images_style = await response_style.json();

    // Hiển thị 3 hình đầu tiên cho galleryContainer1 (content images)
    const firstImages1 = images_content.slice(0, 3); // 3 ảnh đầu tiên
    firstImages1.forEach(img => {
        const imgElement = document.createElement('img');
        imgElement.src = img;
        imgElement.alt = img;
        imgElement.classList.add('img_sample');
        galleryContainer1.appendChild(imgElement);
    });

    // Các hình còn lại ẩn đi cho galleryContainer1
    const remainingImages1 = images_content.slice(3); // Các ảnh còn lại
    remainingImages1.forEach(img => {
        const imgElement = document.createElement('img');
        imgElement.src = img;
        imgElement.alt = img;
        imgElement.classList.add('img_sample', 'hidden');
        galleryContainer1.appendChild(imgElement);
    });

    // Hiển thị 3 hình đầu tiên cho galleryContainer2 (style images)
    const firstImages2 = images_style.slice(0, 3); // 3 ảnh đầu tiên
    firstImages2.forEach(img => {
        const imgElement = document.createElement('img');
        imgElement.src = img;
        imgElement.alt = img;
        imgElement.classList.add('img_sample');
        galleryContainer2.appendChild(imgElement);
    });

    // Các hình còn lại ẩn đi cho galleryContainer2
    const remainingImages2 = images_style.slice(3); // Các ảnh còn lại
    remainingImages2.forEach(img => {
        const imgElement = document.createElement('img');
        imgElement.src = img;
        imgElement.alt = img;
        imgElement.classList.add('img_sample', 'hidden');
        galleryContainer2.appendChild(imgElement);
    });

    // Khi bấm See More cho galleryContainer1 (content images)
    seeMoreBtn1.addEventListener('click', (event) => {
        event.preventDefault(); // Ngăn chặn hành động mặc định của nút
        document.querySelectorAll('.started_2_flex .img_sample.hidden').forEach(img => {
            img.classList.remove('hidden'); // Hiển thị các ảnh còn lại
        });
        seeMoreBtn1.style.display = 'none'; // Ẩn nút See More
        hideBtn1.classList.remove('hidden'); // Hiển thị nút Hide
    });

    // Khi bấm Hide cho galleryContainer1 (content images)
    hideBtn1.addEventListener('click', (event) => {
        event.preventDefault(); // Ngăn chặn hành động mặc định của nút
        document.querySelectorAll('.started_2_flex .img_sample:not(.hidden)').forEach((img, index) => {
            if (index >= 3) img.classList.add('hidden'); // Ẩn lại các ảnh dư thừa
        });
        hideBtn1.classList.add('hidden'); // Ẩn nút Hide
        seeMoreBtn1.style.display = 'inline'; // Hiển thị nút See More
    });

    // Khi bấm See More cho galleryContainer2 (style images)
    seeMoreBtn2.addEventListener('click', (event) => {
        event.preventDefault(); // Ngăn chặn hành động mặc định của nút
        document.querySelectorAll('.started_3_flex .img_sample.hidden').forEach(img => {
            img.classList.remove('hidden'); // Hiển thị các ảnh còn lại
        });
        seeMoreBtn2.style.display = 'none'; // Ẩn nút See More
        hideBtn2.classList.remove('hidden'); // Hiển thị nút Hide
    });

    // Khi bấm Hide cho galleryContainer2 (style images)
    hideBtn2.addEventListener('click', (event) => {
        event.preventDefault(); // Ngăn chặn hành động mặc định của nút
        document.querySelectorAll('.started_3_flex .img_sample:not(.hidden)').forEach((img, index) => {
            if (index >= 3) img.classList.add('hidden'); // Ẩn lại các ảnh dư thừa
        });
        hideBtn2.classList.add('hidden'); // Ẩn nút Hide
        seeMoreBtn2.style.display = 'inline'; // Hiển thị nút See More
    });
});
