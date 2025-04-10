async function uploadImage() {
  const fileInput = document.getElementById('imageUpload');
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select an image!");
    return;
  }
  const formData = new FormData();
  formData.append('file', file);

  const inputImage = document.getElementById('inputImage');
  inputImage.src = URL.createObjectURL(file);

  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();

    const outputImage = document.getElementById('outputImage');
    outputImage.src = `data:image/jpeg;base64,${data.processed_image}`;

    const resultDiv = document.getElementById('predictionResult');

    const tankDetected = data.predictions.some(prediction => prediction.label.toLowerCase() === 'tank');

    if (tankDetected) {
      resultDiv.innerHTML = "Tank Detected!";
      resultDiv.style.color = "green";
    } else {
      resultDiv.innerHTML = "No Tank Detected!";
      resultDiv.style.color = "red";
    }
  } catch (error) {
    console.error('Error uploading image:', error);
    alert("An error occurred while uploading the image.");
  }
}

