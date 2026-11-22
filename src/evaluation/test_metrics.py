from src.evaluation.metrics import (
    calculate_iou,
    match_detections,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    calculate_detection_metrics,
)


def test_iou():
    value = calculate_iou(
        [0, 0, 100, 100],
        [50, 50, 150, 150]
    )

    assert abs(value - 0.142857) < 0.0001


def test_identical_iou():
    value = calculate_iou(
        [0, 0, 100, 100],
        [0, 0, 100, 100]
    )

    assert value == 1.0


def test_no_overlap():
    value = calculate_iou(
        [0, 0, 10, 10],
        [20, 20, 30, 30]
    )

    assert value == 0.0


def test_tp_fp_fn():

    ground_truth = [
        [0, 0, 100, 100],
        [200, 200, 300, 300],
    ]

    predictions = [
        [0, 0, 100, 100],
        [500, 500, 600, 600],
    ]

    confidences = [
        0.95,
        0.80,
    ]

    tp, fp, fn = match_detections(
        ground_truth,
        predictions,
        confidences,
        iou_threshold=0.5
    )

    assert tp == 1
    assert fp == 1
    assert fn == 1


def test_metrics():

    precision = calculate_precision(1, 1)
    recall = calculate_recall(1, 1)
    f1 = calculate_f1(
        precision,
        recall
    )

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


if __name__ == "__main__":

    test_iou()
    test_identical_iou()
    test_no_overlap()
    test_tp_fp_fn()
    test_metrics()

    print("Все тесты пройдены успешно.")