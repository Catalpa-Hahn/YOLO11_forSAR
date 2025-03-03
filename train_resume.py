from ultralytics import YOLO

# 加载部分训练的模型
model = YOLO('./runs/detect/train4/weights/last.pt')

# 恢复训练
results = model.train(resume=True)