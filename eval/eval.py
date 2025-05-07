import re
import json
import argparse
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def extract_ratings(text):
    overall_match = re.search(r'Overall Quality:\s*([\d\.]+)', text)
    confidence_match = re.search(r'Review Confidence:\s*([\d\.]+)', text)
    overall = float(overall_match.group(1)) if overall_match else None
    confidence = float(confidence_match.group(1)) if confidence_match else None
    return overall, confidence

def evaluate_ratings(generated_path):
    preds_oq, preds_conf = [], []
    trues_oq, trues_conf = [], []

    with open(generated_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            pred_text = item.get('predict')
            label_text = item.get('label')

            if pred_text is None or label_text is None:
                continue

            pred_overall, pred_confidence = extract_ratings(pred_text)
            true_overall, true_confidence = extract_ratings(label_text)

            if pred_overall is not None and pred_confidence is not None and true_overall is not None and true_confidence is not None:
                preds_oq.append(pred_overall)
                trues_oq.append(true_overall)
                preds_conf.append(pred_confidence)
                trues_conf.append(true_confidence)

    results = {}

    for name, pred, true in [
        ('Overall Quality', preds_oq, trues_oq),
        ('Review Confidence', preds_conf, trues_conf)
    ]:
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        rmse = mse ** 0.5
        corr, _ = pearsonr(true, pred)

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "Pearson": corr,
        }

    return results

def load_nlg_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return {
        "BLEU-4": metrics.get("predict_bleu-4"),
        "ROUGE-1": metrics.get("predict_rouge-1"),
        "ROUGE-2": metrics.get("predict_rouge-2"),
        "ROUGE-L": metrics.get("predict_rouge-l"),
        "Samples/sec": metrics.get("predict_samples_per_second"),
        "Steps/sec": metrics.get("predict_steps_per_second"),
        "Predict Runtime (s)": metrics.get("predict_runtime")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True, help='Path to generated_predictions.jsonl')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save evaluation results')
    parser.add_argument('--metrics', type=str, default=None, help='Path to NLG metrics .json')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_csv_path = os.path.join(args.save_dir, "summary.csv")
    save_json_path = os.path.join(args.save_dir, f"{args.model_name}_ratings_eval.json")

    rating_results = evaluate_ratings(args.generated)
    nlg_results = {}
    if args.metrics:
        nlg_results = load_nlg_metrics(args.metrics)

    # 汇总一行（用于csv）
    row = {
        "Model": args.model_name,
        "OQ_MAE": rating_results['Overall Quality']['MAE'],
        "OQ_RMSE": rating_results['Overall Quality']['RMSE'],
        "OQ_Pearson": rating_results['Overall Quality']['Pearson'],
        "Conf_MAE": rating_results['Review Confidence']['MAE'],
        "Conf_RMSE": rating_results['Review Confidence']['RMSE'],
        "Conf_Pearson": rating_results['Review Confidence']['Pearson'],
    }
    row.update(nlg_results)

    # 保存summary.csv，只添加新模型
    if os.path.exists(save_csv_path):
        df = pd.read_csv(save_csv_path)
        if args.model_name not in df['Model'].values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(save_csv_path, index=False)
            print(f"✅ Added {args.model_name} to {save_csv_path}")
        else:
            print(f"⚠️ Model {args.model_name} already exists in {save_csv_path}, skipping.")
    else:
        df = pd.DataFrame([row])
        df.to_csv(save_csv_path, index=False)
        print(f"✅ Created {save_csv_path}")

    # 保存详细单独json（包括BLEU/ROUGE）
    detailed_result = {
        "Overall Quality": rating_results['Overall Quality'],
        "Review Confidence": rating_results['Review Confidence'],
        "NLG Metrics": nlg_results,
    }
    with open(save_json_path, "w") as f:
        json.dump(detailed_result, f, indent=2)
    print(f"✅ Saved detailed evaluation to {save_json_path}")
