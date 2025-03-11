const video = document.getElementById('webcam');
const emotionText = document.getElementById('emotion');

// Truy cập webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("Webcam access denied:", err);
    emotionText.textContent = "Cannot access webcam.";
  });

// Tạo canvas để chụp ảnh từ video
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

// Hàm gửi ảnh về server
async function sendFrameToServer() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      emotionText.textContent = `Emotion: ${data.emotion}`;
    } catch (err) {
      console.error("Failed to predict emotion:", err);
    }
  }, 'image/jpeg');
}

// Gửi frame mỗi 1000ms (1 giây)
setInterval(sendFrameToServer, 1000);
