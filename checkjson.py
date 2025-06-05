import json
import numpy as np

def resample_polygon(points, num_points):
    points = np.array(points)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumulative[-1]
    desired = np.linspace(0, total_length, num_points+1)[:-1]
    resampled = []
    for d in desired:
        idx = np.searchsorted(cumulative, d) - 1
        idx = min(idx, len(points)-2)
        t = (d - cumulative[idx]) / (cumulative[idx+1] - cumulative[idx]) if cumulative[idx+1] > cumulative[idx] else 0
        pt = (1-t)*points[idx] + t*points[idx+1]
        resampled.append(pt.tolist())
    return resampled

# Đọc file json
with open('BTXRD/images/annotations/IMG000024.json', encoding='utf-8') as f:
    data = json.load(f)

# Số điểm polygon model trả về
num_points = 1047  # (polygon_len // 2 từ model.py)

for shape in data['shapes']:
    if shape.get('shape_type', 'polygon') == 'polygon' and len(shape['points']) > 2:
        shape['points'] = resample_polygon(shape['points'], num_points)

# Ghi lại file json (nếu muốn)
with open('BTXRD/images/annotations/IMG000024_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)