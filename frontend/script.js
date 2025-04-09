async function uploadImage() {
  const fileInput = document.getElementById('imageUpload');
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select an image!")
    return;
  }
  const formData = new FormData();
  formData.append('file', file);

  const imagePreview = document.getElementById('imagePreview');
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  imagePreview.innerHTML = '';
  imagePreview.appendChild(img);

  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    const resultDiv = document.getElementById('predictionResult');

    if (data.prediction == 'Tank') {
      resultDiv.innerHTML = "Tank Detected!";
      resultDiv.style.color = "green";
    } else {
      resultDiv.innerHTML = "No Tank Detected!";
      resultDiv.style.color = "red";
    }
  } catch (error) {
    console.error('Error uploading image:', error)
    alert("An error occurred while uploading the image.");
  }
}
