# Model Evaluation Notes

Fraud detection is an imbalanced classification problem, so accuracy alone is not a useful production metric.

## Primary metrics

**Precision** measures how many transactions flagged as fraud are actually fraudulent. Higher precision reduces unnecessary investigation workload.

**Recall** measures how many actual fraudulent transactions are detected. Higher recall reduces missed fraud.

**F1 score** provides a single measure that balances precision and recall.

**ROC-AUC** measures ranking performance across classification thresholds, but it should be interpreted alongside precision-recall metrics for highly imbalanced datasets.

## Threshold selection

The default probability threshold is not automatically the best operating point. The project evaluates the precision-recall tradeoff and selects a threshold based on the desired balance between catching fraud and limiting false alerts.

## Production considerations

A production fraud system should monitor more than offline model metrics. Useful operational measures include:

- Fraud capture rate
- False-positive rate
- Alert volume
- Review workload
- Prediction latency
- Feature drift
- Prediction drift

Thresholds should be revisited when transaction patterns, fraud behavior, or business costs change.
