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
