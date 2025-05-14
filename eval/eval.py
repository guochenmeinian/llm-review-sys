import re
import json
import argparse
import numpy as np
import os
import glob
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import evaluate

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
            pred_text = item.get('generated_output') or item.get('predict')
            label_text = item.get('output') or item.get('label')

            if pred_text is None or label_text is None:
                continue

            pred_overall, pred_confidence = extract_ratings(pred_text)
            true_overall, true_confidence = extract_ratings(label_text)

            if None not in (pred_overall, pred_confidence, true_overall, true_confidence):
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
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Pearson": round(corr, 4),
        }

    return results

def evaluate_nlg_metrics(generated_path):
    with open(generated_path, 'r') as f:
        data = [json.loads(line) for line in f]

    references = [item["output"] for item in data if item.get("output")]
    predictions = [item.get("generated_output") or item.get("predict") for item in data if item.get("generated_output") or item.get("predict")]

    if len(predictions) != len(references) or len(predictions) == 0:
        print("⚠️ NLG评估数据不完整，跳过NLG指标")
        return {}

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])["bleu"]

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return {
        "BLEU-4": round(bleu_score * 100, 2),
        "ROUGE-1": round(rouge_score["rouge1"] * 100, 2),
        "ROUGE-2": round(rouge_score["rouge2"] * 100, 2),
        "ROUGE-L": round(rouge_score["rougeL"] * 100, 2),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name, e.g., full_context_qlora_dpo')
    parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save evaluation results')
    parser.add_argument('--metrics', type=str, default=None, help='Optional: path to NLG metrics .json')
    args = parser.parse_args()

    # search JSONL file
    jsonl_candidates = glob.glob(f"eval/{args.model_name}.jsonl") + \
                       glob.glob(f"results_eval_{args.model_name}.jsonl")
    if not jsonl_candidates:
        raise FileNotFoundError(f"❌ Cannot find results_eval_{args.model_name}.jsonl in ./ or ./eval/")
    generated_path = jsonl_candidates[0]
    print(f"📄 Using generated file: {generated_path}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_csv_path = os.path.join(args.save_dir, "summary.csv")
    save_json_path = os.path.join(args.save_dir, f"{args.model_name}_ratings_eval.json")

    # rating and generation evaluation
    rating_results = evaluate_ratings(generated_path)
    nlg_results = evaluate_nlg_metrics(generated_path)

    # summary for summary.csv
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

    # save summary.csv (incrementally)
    if os.path.exists(save_csv_path):
        df = pd.read_csv(save_csv_path)
        if args.model_name not in df['Model'].values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(save_csv_path, index=False)
            print(f"✅ Added {args.model_name} to {save_csv_path}")
        else:
            print(f"⚠️ Model {args.model_name} already exists in {save_csv_path}, skipping.")
    else:
        pd.DataFrame([row]).to_csv(save_csv_path, index=False)
        print(f"✅ Created {save_csv_path}")

    # save detailed JSON results
    detailed_result = {
        "Overall Quality": rating_results['Overall Quality'],
        "Review Confidence": rating_results['Review Confidence'],
        "NLG Metrics": nlg_results,
    }
    with open(save_json_path, "w") as f:
        json.dump(detailed_result, f, indent=2)
    print(f"✅ Saved detailed evaluation to {save_json_path}")
