import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.metrics import (
    analyze_errors,
    calculate_detection_metrics,
    calculate_map,
)
from src.evaluation.plots import plot_metrics
from src.evaluation.robustness import create_variants


def get_demo_data() -> Dict[str, Any]:
    """Демо-данные для проверки скрипта до получения предсказаний от моделей."""
    return {
        "model_name": "demo_baseline",
        "iou_threshold": 0.5,
        "ground_truth": [
            [50.0, 60.0, 180.0, 320.0],
            [240.0, 90.0, 350.0, 310.0],
        ],
        "predictions": [
            [55.0, 62.0, 178.0, 318.0],
            [250.0, 100.0, 340.0, 290.0],
            [420.0, 120.0, 500.0, 260.0],
        ],
        "confidences": [0.92, 0.78, 0.45],
    }


def load_eval_input(input_path: Optional[str]) -> Dict[str, Any]:
    """Загружает входные данные из JSON или возвращает демо-данные."""
    if not input_path:
        return get_demo_data()

    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["ground_truth", "predictions", "confidences"]
    for key in required:
        if key not in data:
            raise KeyError(f"В файле {input_path} отсутствует обязательное поле '{key}'.")

    return data


def run_evaluation(
    input_path: Optional[str] = None,
    image_path: Optional[str] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Запускает полный цикл оценки, искажений (опционально) и графиков."""
    data = load_eval_input(input_path)

    model_name: str = data.get("model_name", "model")
    iou_threshold: float = float(data.get("iou_threshold", 0.5))
    ground_truth: List[List[float]] = data["ground_truth"]
    predictions: List[List[float]] = data["predictions"]
    confidences: List[float] = [float(c) for c in data["confidences"]]

    det_metrics = calculate_detection_metrics(
        ground_truth=ground_truth,
        predictions=predictions,
        confidences=confidences,
        iou_threshold=iou_threshold,
    )
    map_metrics = calculate_map(
        ground_truth=ground_truth,
        predictions=predictions,
        confidences=confidences,
    )
    error_analysis = analyze_errors(
        ground_truth=ground_truth,
        predictions=predictions,
        confidences=confidences,
        iou_threshold=iou_threshold,
    )

    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics"
    plots_dir = out_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    chart_path = plots_dir / f"{model_name}_metrics.png"
    metric_names = ["Precision", "Recall", "F1", "AP50", "AP75", "mAP50:95"]
    metric_values = [
        det_metrics["precision"],
        det_metrics["recall"],
        det_metrics["f1"],
        map_metrics["AP50"],
        map_metrics["AP75"],
        map_metrics["mAP50:95"],
    ]
    plot_metrics(
        names=metric_names,
        values=metric_values,
        title=f"Метрики детекции ({model_name})",
        output_path=str(chart_path),
    )

    robustness_output: Optional[str] = None
    if image_path:
        rob_dir = out_root / "robustness" / Path(image_path).stem
        create_variants(image_path, str(rob_dir))
        robustness_output = str(rob_dir)

    report: Dict[str, Any] = {
        "model_name": model_name,
        "detection_metrics": det_metrics,
        "map_metrics": {
            "AP50": map_metrics["AP50"],
            "AP75": map_metrics["AP75"],
            "mAP50:95": map_metrics["mAP50:95"],
            "per_threshold": {
                str(k): v for k, v in map_metrics["per_threshold"].items()
            },
        },
        "error_analysis": error_analysis,
        "artifacts": {
            "metrics_chart": str(chart_path),
            "robustness_dir": robustness_output,
        },
    }

    report_path = metrics_dir / f"{model_name}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report["artifacts"]["report_json"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Единый скрипт оценки качества детекции, искажений и графиков."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Путь к JSON-файлу с ground_truth, predictions и confidences.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Путь к изображению для генерации вариантов искажений (опционально).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Корневая папка результатов (по умолчанию results).",
    )
    args = parser.parse_args()

    report = run_evaluation(
        input_path=args.input,
        image_path=args.image,
        output_dir=args.output_dir,
    )

    det = report["detection_metrics"]
    mp = report["map_metrics"]
    err = report["error_analysis"]["confusion_counts"]

    print(f"Модель: {report['model_name']}")
    print(
        f"TP={int(det['TP'])}, FP={int(det['FP'])}, FN={int(det['FN'])} | "
        f"Precision={det['precision']:.4f}, Recall={det['recall']:.4f}, F1={det['f1']:.4f}"
    )
    print(
        f"AP50={mp['AP50']:.4f}, AP75={mp['AP75']:.4f}, mAP50:95={mp['mAP50:95']:.4f}"
    )
    print(
        f"Ошибки FP: low_iou_or_dup={err['FP_low_iou_or_dup']}, "
        f"background={err['FP_background']}"
    )
    print(f"Отчёт сохранён: {report['artifacts']['report_json']}")
    print(f"График сохранён: {report['artifacts']['metrics_chart']}")
    if report["artifacts"]["robustness_dir"]:
        print(f"Искажения сохранены в: {report['artifacts']['robustness_dir']}")


if __name__ == "__main__":
    main()