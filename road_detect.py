# -*- coding: utf-8 -*-
"""
@File ：detect.py
@IDE ：PyCharm

"""

import time
from ultralytics import YOLO
import cv2
from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLO 模型
    model = YOLO(model=r'')
    # 输入视频路径和输出视频路径
    input_video_path = r''
    output_video_path = r''
    start_time = time.time()  # 记录开始时间
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开输入视频")
        exit()

    # 获取输入视频的属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

    # 定义视频编解码器和输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"开始处理视频，总帧数: {frame_count}, 输出分辨率: {width}x{height}")

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 结束读取视频

        # 使用 YOLO 模型预测当前帧
        results = model.predict(frame, save=False, iou=0.1)

        # 获取带标注的帧
        result_frame = results[0].plot()

        # 写入处理后的帧到输出视频
        out.write(result_frame)

        frame_id += 1
        # if frame_id % 100 == 0:
        print(f"已处理帧数: {frame_id}")

    # 释放资源
    cap.release()
    out.release()
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    # print("视频处理完成，输出文件保存在:", output_video_path)
    print(f"总处理时间: {total_time:.2f} 秒，平均每帧时间: {total_time / frame_id:.4f} 秒")
    print(f"总处理时间: {total_time:.2f} 秒，平均每帧时间: {frame_id / total_time:.4f} 秒")

