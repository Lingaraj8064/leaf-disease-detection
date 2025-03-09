document.getElementById("upload-form").addEventListener("submit", async function(event) {
    event.preventDefault();
    
    let formData = new FormData();
    let fileInput = document.getElementById("file-input").files[0];
    
    if (!fileInput) {
        alert("Please select an image to upload.");
        return;
    }

    formData.append("file", fileInput);

    try {
        let response = await fetch("/", {
            method: "POST",
            body: formData
        });

        let result = await response.json();

        document.getElementById("uploaded-image").src = result.image_url;
        document.getElementById("uploaded-image").style.display = "block";
        document.getElementById("prediction-text").innerText = "Prediction: " + result.prediction;
    } catch (error) {
        console.error("Error:", error);
    }
});
