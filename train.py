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

    model = YOLO('yolov8s-seg.pt')

    model.train(
        data='BTXRD.yaml',
        epochs=100,
        imgsz=768,
        batch=8,  # Giảm batch để phù hợp VRAM
        optimizer='AdamW',
        patience=50,
        name='btxrd_yolov8s_seg',
        device=0,
        cache='disk',
        close_mosaic=20,
        dropout=0.3,
        freeze=10,
        amp=True,
        autoanchor=True,
        multi_scale=True,
        seg_loss_gain=1.5,
        **filtered
    )

if __name__ == '__main__':
    main()