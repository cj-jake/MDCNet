# -*- coding: utf-8 -*-
"""
@File ：detect.py
@IDE ：PyCharm
"""

from ultralytics import YOLO

if __name__ == '__main__':


    # Load a model
    model = YOLO(model=r'')
    model.predict(source=r'data/image',
                                save=True,
                                show=False,
                                # show_boxes=True,  # 显示边框
                                # show_labels=False,  # 不显示标签
                                show_conf=False, # 不显示置信度分数
                                # stream=True,  # 流式处理，逐帧加载
                                iou = 0.1  # 设置 IoU 阈值，调小可减少重叠框
                  )





