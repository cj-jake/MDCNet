# MDCNet
1.Comparison effect of real road testing


<div class="video-container">
  <video autoplay muted loop>
       <source src="https://github.com/cj-jake/MDCNet/raw/refs/heads/master/videos/paper05.mp4" type="video/mp4">
  </video>
</div>
<style>
  /* 重置页面默认样式 */
  html, body {
    margin: 0;
    padding: 0;
    overflow: hidden;
  }
  /* 外层容器设置 */
  .video-container {
    position: fixed;
    top: 50%;
    left: 50%;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    transform: translate(-50%, -50%);
    z-index: -1;
  }
  /* 视频设置 */
  .video-container video {
    position: absolute;
    top: 50%;
    left: 50%;
    min-width: 100%;
    min-height: 100%;
    width: auto;
    height: auto;
    transform: translate(-50%, -50%);
    object-fit: cover;
    /* 如果背景不需要黑色，可以去掉 background-color */
    background-color: transparent;
  }
</style>


