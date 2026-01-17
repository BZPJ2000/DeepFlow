from sklearn.metrics import average_precision_score

def mean_average_precision(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='macro')

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

# Example usage
if __name__ == "__main__":
    y_true = [1, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.8, 0.2]

    print(f"mAP: {mean_average_precision(y_true, y_pred):.4f}")
    print(f"IoU: {iou([0, 0, 10, 10], [5, 5, 15, 15]):.4f}")