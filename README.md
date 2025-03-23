# MDCNet
1.Comparison effect of real road testing


<div class="video-container">
  <div class="video-wrapper">
    <iframe 
      src="https://cj-jake.github.io/MDCNet/videos/paper05.mp4" allowfullscreen>
    </iframe>
  </div>
</div>

<style>
  /* 背景渐变，让整体 UI 更现代 */
  .video-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh; /* 让视频垂直居中 */
    background: linear-gradient(135deg, #1e3c72, #2a5298); /* 蓝色渐变背景 */
  }

  /* 视频外框，带阴影和动画 */
  .video-wrapper {
    position: relative;
    width: 80%;
    max-width: 900px;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }

  /* 鼠标悬停时增加放大效果 */
  .video-wrapper:hover {
    transform: scale(1.05);
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5);
  }

  iframe {
    width: 100%;
    height: 500px;
    border: none;
  }

  /* 响应式适配 */
  @media (max-width: 768px) {
    .video-wrapper {
      width: 95%;
    }

    iframe {
      height: 300px;
    }
  }
</style>


