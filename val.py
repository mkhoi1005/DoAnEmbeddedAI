from ultralytics import YOLO

def main():
    # Load mô hình đã huấn luyện
    model = YOLO("btxrd_yolov8n_seg_new1/content/DoAnEmbeddedAI/runs/segment/btxrd_yolov8n_seg/weights/best.pt")

    # Đánh giá trên tập validation
    metrics = model.val(data="BTXRD.yaml")
    print(metrics)

if __name__ == '__main__':
    main()
