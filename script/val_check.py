from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("D:/EmbededAI/runs/segment/btxrd_seg_train5/weights/best.pt")
    results = model.val(data="D:/EmbededAI/datasets/BTXRD/BTXRD.yaml", task="segment")
    print(results)
