# -*- coding: utf-8 -*-
"""
@Auth ：
@File ：trian.py

"""
import warnings
import torch

from ultralytics.utils.benchmarks import benchmark

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    # 释放 PyTorch 的 GPU 缓存显存
    # torch.cuda.empty_cache()

    model = YOLO(model=r'ultralytics/cfg/models/11/yolo11_EGS_NoNeck.yaml')
    model.train(data=r'roadRDD2023.yaml',
                imgsz=640,
                epochs=600,
                batch=16,
                workers=0,
                device='',
                patience=100,
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/road/train',
                name='exp',
                single_cls=False,
                cache=False,
                verbose=True,
                augment = True
                )
    model.val(data=r'roadRDD2023.yaml',)
    model.val(data=r'roadRDD2023.yaml',split='test')
    #执行验证

    # model.val(data=r'roadRDD2023.yaml', )
    # Benchmark on GPU
    #benchmark(model=model, data="roadRDD2023.yaml", imgsz=640, half=False)



