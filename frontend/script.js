async function uploadFile() {
  const fileInput = document.getElementById("fileUpload");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a file!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const inputImage = document.getElementById("inputImage");
  const outputImage = document.getElementById("outputImage");
  const inputGif = document.getElementById("inputGif");
  const outputGif = document.getElementById("outputGif");

  inputImage.style.display = "none";
  outputImage.style.display = "none";
  inputGif.style.display = "none";
  outputGif.style.display = "none";

  const resultDiv = document.getElementById("predictionResult");
  resultDiv.innerHTML = "";
  resultDiv.style.color = "white";

  const fileURL = URL.createObjectURL(file);
  if (file.type.startsWith("image/")) {
    inputImage.src = fileURL;
    inputImage.style.display = "block";
  } else if (file.type.startsWith("video/")) {
    inputGif.src = ""; // очищення
    inputGif.style.display = "none";
  }


  try {
    const response = await fetch("http://api.local:8081/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (file.type.startsWith("image/")) {
      outputImage.src = `data:image/jpeg;base64,${data.processed_image}`;
      outputImage.style.display = "block";

      const tankDetected = data.predictions.some(
        (prediction) => prediction.label.toLowerCase() === "tank"
      );
      resultDiv.innerHTML = tankDetected ? "Tank Detected!" : "No Tank Detected!";
      resultDiv.style.color = tankDetected ? "green" : "red";
    } else if (file.type.startsWith("video/")) {
      const inputUrl = `data:image/gif;base64,${data.input_gif}`;
      const outputUrl = `data:image/gif;base64,${data.output_gif}`;

      inputGif.src = "";
      outputGif.src = "";
      setTimeout(() => {
        inputGif.src = inputUrl;
        outputGif.src = outputUrl;
        inputGif.style.display = "block";
        outputGif.style.display = "block";
      }, 1);

      resultDiv.innerHTML = data.has_tank ? "Tank Detected in Video!" : "No Tank Found in Video!";
      resultDiv.style.color = data.has_tank ? "green" : "red";
    }
  } catch (error) {
    console.error("Error uploading file:", error);
    alert("An error occurred while uploading the file.");
  }
}

