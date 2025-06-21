# Evaluating-Model-Explanations-without-Ground-Truth

This repository contains code accompanying the FAccT 2025 paper "Evaluating Model Explanations without Ground Truth". It implements AXE: a ground-truth Agnostic eXplanation Evaluation framework. This provides the first principled strategy to meausure the quality of local feature-importance explanations such as those output by SHAP, LIME, or Integrated-Gradients. AXE is unique because it does not evaluate explanation quality by comparing real explanations with ideal explanations. In practical scenarios lacking access to an oracle that can produce ideal explanations to compare with, AXE can be use seamlessly to measure explanation quality.

Please see the [paper](./Paper/preprint.pdf) for details, or the accompanying [slides](./Paper/FAccT25-Slideshow.pdf) for a quick overview.

## Three Principles for Evaluating Model Explanations

In this paper, we propose three foundational principles necessary to evaluate all local feature importance explanations.

1. **Local Contextualization**: Explanations should depend on the datapoint they seek to explain.
2. **Model Relativism**: Explanations should depend on the model they seek to explain.
3. **On-Manifold Evaluation**: Explanations should be indedependent of off-point model behavior.

Surprisingly, no previous evaluation strategies satisfy all three of these basic principles. AXE satisfies all three of these principles, and has the added benefit of not needing access to ground truth explanations.

## AXE: A ground-truth Agnostic eXplanation Evaluation framework

AXE operationalises the idea that the highest quality explanation must be one that helps the user predict the model output. In accordance with the 3 foundational principles above, AXE constructs a unique KNN model per datapoint explained, using only a subset of the most important features as determined by the explanation. The accuracy of this model is used to infer the quality of the explanation.

An implementation using [FAISS](https://github.com/facebookresearch/faiss) can be found in [Sec 4.2](./Sec4.2-BaselineComparisons-Figure6andFigure7). This is recommended for high throughput applications where the use-case requires efficiency.

An implementation using exact KNN models from sklearn can be found in [Sec 4.1](./Sec4.1-DetectingFairwashing-Table2). This is recommended for specialised applications where the use-case requires accuracy.

## Repository Structure

The repository contains all the code needed to replicate the experiments in the paper. The explanation disagreement example and illustrative example comparing with PGI are self contained in subdirectories [Sec1](./Sec1-ExplanationDisagreement-Figure1) and [Sec3](Sec3.2-IllustrativeAxeExample-Figure4andFigure5) respectively. The code for the fairwashing adversarial attack experiment in [Sec 4.1](./Sec4.1-DetectingFairwashing-Table2) and the comparison with OpenXAI baselines in [Sec 4.2](./Sec4.2-BaselineComparisons-Figure6andFigure7) is included as submodules.

## Contact

If you have any issues with the code please open a GitHub issue. For any direct suggestions, errors, or improvements please contact first author [Kai Rawal](mailto:kaivalya.rawal@oii.ox.ac.uk). Please address any other feedback or comments using the author emails provided in the paper.

## Citation

If you find this paper useful or use AXE in your research, please cite us using the citation below.

```
TBD
```


## Licence

All code is released under the MIT License.

