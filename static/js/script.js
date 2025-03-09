document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("imagePreview");

    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewContainer.innerHTML = `<img src="${e.target.result}" class="img-thumbnail mt-2" style="max-width: 200px;">`;
            };
            reader.readAsDataURL(file);
        }
    });
});
