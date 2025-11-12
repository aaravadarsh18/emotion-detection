# Model Comparison Summary

| Model | Macro F1 | Accuracy | Notes |
|--------|-----------|----------|-------|
| TF-IDF + LogReg | 0.62 | 0.67 | Simple, interpretable baseline. |
| RoBERTa (TweetEval) | 0.79 – 0.81 | 0.83 | Contextual model, best performance. |

### Observations
- Transformer model improves minority-class (*optimism*) recall by ~25%.
- TF-IDF + LR performs competitively with far less computation, showing strong lexical separability.
- Error analysis shows overlap between *anger* and *sadness* due to similar affective words.
- Linguistic features (emoji density, tweet length) correlate weakly with emotion intensity.
