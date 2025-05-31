import yaml
from ultralytics import YOLO

def main():
    with open('hyp.yaml', 'r', encoding='utf-8') as f:
        hyp_args = yaml.safe_load(f)

    valid_keys = {
        'lr0', 'lrf', 'momentum', 'weight_decay',
        'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
        'box', 'cls', 'dfl', 'seg',
        'hsv_h', 'hsv_s', 'hsv_v',
        'degrees', 'translate', 'scale', 'shear', 'perspective',
        'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste'
    }
    filtered = {k: v for k, v in hyp_args.items() if k in valid_keys}

    model = YOLO('best.pt')  # mô hình segmentation nhẹ

    model.train(
        data='BTXRD.yaml',
        epochs=200,                 # tăng số epoch do mô hình nhỏ học chậm hơn
        imgsz=640,
        batch=16,
        optimizer='AdamW',
        patience=80,               # tăng patience để không dừng quá sớm
        name='btxrd_yolov8n_seg',
        device=0,
        cache=True,
        close_mosaic=30,           # mở rộng thời gian dùng mosaic
        dropout=0.3,               # giảm nhẹ dropout phù hợp mô hình nhỏ
        freeze=0,                  # đóng băng ít layer hơn do model nhỏ
        amp=True,
        cos_lr=True,
        **filtered
    )

if __name__ == '__main__':
    main()
