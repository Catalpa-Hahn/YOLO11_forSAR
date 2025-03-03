import subprocess
import platform
import pathlib
from ultralytics import YOLO

def main():
    models = ['best.pt']#['best_500epochs.pt', 'best_600epochs.pt']
    # 选择dataset文件
    plt = platform.system() # 判断操作系统类型
    if plt == 'Windows':
        dataset_path = './data/SARDet-100K_local.yaml'
    elif plt == 'Linux':
        dataset_path = './data/SARDet-100K.yaml'
    else:
        raise Exception('Unsupported platform')

    # 运行测试
    for i in range(len(models)):
        print('正在测试模型{}...'.format(models[i]))

        # 加载训练好的模型
        model = YOLO("./runs/detect/continue_22/weights/{}".format(models[i]))  # 替换为你的模型路径

        # 测试模型
        results = model.val(
            task='detect',
            mode='test',
            data=dataset_path,  # 数据配置文件路径
            batch=64,  # 批量大小
            imgsz=512,  # 输入图像大小
            conf=0.001,  # 目标检测的置信度阈值
            iou=0.6,  # NMS 的 IoU 阈值
            device='',  # 使用的设备，默认为空（自动选择）
            split='test',  # 使用测试集进行验证
            save_json=False,  # 是否将结果保存为 JSON 文件
            save_hybrid=False,  # 是否保存混合版本的标签
            plots=True,  # 是否保存图表
            half=False,  # 是否使用半精度（FP16）
            dnn=False,  # 是否使用 OpenCV DNN 进行 ONNX 推理
        )

        # 打印测试结果
        # print("测试结果:")
        # print(f"mAP@0.5: {results.box.map}")  # mAP@0.5
        # print(f"mAP@0.5:0.95: {results.box.map50}")  # mAP@0.5:0.95
        # print(f"精确率 (Precision): {results.box.precision}")
        # print(f"召回率 (Recall): {results.box.recall}")

# print('测试结束！')

if __name__ == '__main__':
    main()