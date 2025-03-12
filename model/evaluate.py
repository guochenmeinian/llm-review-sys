from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, test_dataset):
    """评估模型性能"""
    predictions = []
    references = []
    
    for sample in test_dataset:
        output = model.generate(sample["input"])
        predictions.append(output)
        references.append(sample["reference"])
    
    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="macro")
    
    return {
        "accuracy": accuracy,
        "f1_score": f1
    }