# Distilling CLIP Zero-Shot Capability into ResNet

## Summary
Exploring how to transfer zero-shot image classification from a large CLIP vision–language model into a compact ResNet backbone for faster, lighter deployment.

## Why
- Smaller model, lower latency, easier edge / batch deployment.
- Keeps prompt-based zero-shot classification without full VLM overhead.

## Distillation Strategies (Under Exploration)
- Final feature regression (student matches teacher image embedding).
- Intermediate layer alignment (multi-stage feature guidance).
- Contrastive + logit distillation (class/text-aware similarity shaping).
- Masked / partial feature prediction (robustness and regularization).

## Use
1. Run chosen distillation script.
2. Cache text embeddings once.
3. Perform zero-shot classification with distilled student.

## Status
Active experiments comparing accuracy vs cost across strategies.