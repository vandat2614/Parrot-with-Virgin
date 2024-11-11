document.addEventListener("DOMContentLoaded", function () {
    const styleDropzone = document.getElementById("style-dropzone");
    const contentDropzone = document.getElementById("content-dropzone");
    const styleInput = document.getElementById("style");
    const contentInput = document.getElementById("content");

    function handleFileSelect(input, dropzone) {
        const file = input.files[0]; 
        if (file && file.type.startsWith("image/")) {
            const img = document.createElement("img");
            img.src = URL.createObjectURL(file);
            dropzone.querySelector(".image-box").innerHTML = ''; 
            dropzone.querySelector(".image-box").appendChild(img);
        }
    }

    styleInput.addEventListener("change", function () {
        handleFileSelect(styleInput, styleDropzone);
    });

    contentInput.addEventListener("change", function () {
        handleFileSelect(contentInput, contentDropzone);
    });

    styleDropzone.addEventListener("dragover", function (e) {
        e.preventDefault();
        styleDropzone.classList.add("dragover");
    });

    styleDropzone.addEventListener("dragleave", function () {
        styleDropzone.classList.remove("dragover");
    });

    styleDropzone.addEventListener("drop", function (e) {
        e.preventDefault();
        styleDropzone.classList.remove("dragover");

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            styleInput.files = e.dataTransfer.files;
            handleFileSelect(styleInput, styleDropzone);
        }
    });

    contentDropzone.addEventListener("dragover", function (e) {
        e.preventDefault();
        contentDropzone.classList.add("dragover");
    });

    contentDropzone.addEventListener("dragleave", function () {
        contentDropzone.classList.remove("dragover");
    });

    contentDropzone.addEventListener("drop", function (e) {
        e.preventDefault();
        contentDropzone.classList.remove("dragover");

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            contentInput.files = e.dataTransfer.files;
            handleFileSelect(contentInput, contentDropzone);
        }
    });
});
