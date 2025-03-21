# -*- coding: utf-8 -*-
"""
@Auth ：
@File ：trian.py

"""
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # # model = YOLO(model=r'ultralytics\cfg\models\11\yolo11_GS_ShuffleAttention.yaml')
    # model_11 = YOLO("runs/road/linux/训练600论数/org/weights/best.pt")
    # model_11gs = YOLO("runs/road/linux/训练600论数/GS/weights/best.pt")
    # # model_11Egs = YOLO("runs/road/linux/训练600论数/gs_self_4/weights/best.pt")
    # model__11shuA = YOLO("runs/road/linux/训练600论数/shut/exp9/weights/best.pt")
    # model_11shuA_gs = YOLO("runs/road/linux/训练600论数/gs_shut/weights/best.pt")
    # #
    # model_11.val(data=r'roadRDD2023.yaml',split='test')
    # model_11gs.val(data=r'roadRDD2023.yaml',split='test')
    # # model_11Egs.val(data=r'roadRDD2023.yaml',split='test')
    # model__11shuA.val(data=r'roadRDD2023.yaml',split='test')
    # model_11shuA_gs.val(data=r'roadRDD2023.yaml',split='test')



    #验证论文部分
    # model = YOLO(model=r'ultralytics\cfg\models\11\yolo11_GS_ShuffleAttention.yaml')
    model_11org = YOLO("runs/road/论文6个验证/论文的6个实验_road2023/1_org/exp30/weights/best.pt")
    model_11gs_neck_no = YOLO("runs/road/论文6个验证/论文的6个实验_road2023/2_gs_neck_no/exp41/weights/best.pt")
    model__11shuA = YOLO("runs/road/论文6个验证/论文的6个实验_road2023/3_shuA_完整跑完600轮/exp2/weights/best.pt")
    model_11Egs_neck_no = YOLO("runs/road/论文6个验证/论文的6个实验_road2023/4_修个Egs_neck_no/exp46/weights/best.pt")
    model_11gs_shuA_neck_no = YOLO("runs/road/论文6个验证/论文的6个实验_road2023/5_gs_shut_neck_no/exp37/weights/best.pt")
    model_11our = YOLO("runs/road/论文6个验证/论文的6个实验_road2023/6_0.683_EGS_ShuA/exp46/weights/best.pt")
    #

    model_11org.to("cpu")
    model_11gs_neck_no.to("cpu")
    model__11shuA.to("cpu")
    model_11Egs_neck_no.to("cpu")
    model_11gs_shuA_neck_no.to("cpu")
    model_11our.to("cpu")
    model_11org.val(data=r'roadRDD2023.yaml', split='test')
    model_11gs_neck_no.val(data=r'roadRDD2023.yaml', split='test')
    model__11shuA.val(data=r'roadRDD2023.yaml',split='test')
    model_11Egs_neck_no.val(data=r'roadRDD2023.yaml', split='test')
    model_11gs_shuA_neck_no.val(data=r'roadRDD2023.yaml', split='test')
    model_11our.val(data=r'roadRDD2023.yaml', split='test')

    from ultralytics.utils.benchmarks import benchmark


    #执行验证



