# AI Interaction Log - ViT Traffic Sign Classification

This log tracks my AI interactions and reflections during the Vision Transformer traffic sign project (inference, verification and corruption analysis). It documents:
- The important prompts I used
- How AI suggestions influenced my decisions
- Which suggestions were accepted, adjusted, or rejected, and why
- Failure patterns I observed
- Best practices I learned for using AI effectively

**Project notebooks:** `inference.ipynb`, `confusion_analysis.ipynb`
**Model / data:** `kelvinandreas/vit-traffic-sign-GTSRB` on `tanganke/gtsrb` (43 classes; splits: `test`, `motion_blur`, `gaussian_noise`)

---

## Summary: Failure Patterns & Best Practices

### 1. Failure Patterns to Watch For

- **Plausible but unverified assumptions about labels:**
  Both the model and the dataset have 43 classes, but matching output size does not prove matching class order. Trusting this without a check could silently produce wrong evaluations.
- **Generic or invented preprocessing:**
  AI can suggest preprocessing values that do not belong to the chosen pretrained model. The processor saved with the model (224 × 224, rescale 1/255, mean/std 0.5) is the reliable source.
- **Silent data loss and misleading metrics:**
  Code can run without errors yet drop rows or report raw error counts that ignore class size. Row counts, probability sums and class-relative percentages have to be checked.
- **Overstated explanations:**
  AI readily offers causes (e.g. "brightness caused the errors") that the notebook only supports as descriptive differences, not causal evidence.

---

### 2. Best Practices I Learned

- **Ask one focused question at a time:**
  Give the AI the relevant code, error or result, ask for one specific task, then run it and compare the output with the project requirements.
- **Verify before trusting:**
  Check label names, evaluate a shuffled sample, verify row counts for every split, and confirm confidence equals the maximum softmax score.
- **Report counts and rates together:**
  Larger classes naturally produce more errors, so keep both raw counts and percentage of the true class.
- **Separate measured results from hypotheses:**
  Accuracy and pairwise confusion rates are measurements. Reasons why corruption changes predictions are hypotheses that need further testing.
- **Inspect every figure after rendering:**
  A plot should support the numerical table, not replace it.

---

## Log of AI Interactions

### : Understanding the Task and Planning the Workflow

- **Who:** Rudra
- **Goal:** Break the project into a clear workflow for evaluating a pretrained ViT on clean, motion-blur and Gaussian-noise images without retraining.
- **Prompt (important ones):**
  ```
  I have a pretrained ViT traffic-sign classifier and a GTSRB dataset. I need to evaluate it on clean images, motion blur and Gaussian noise, then analyse its mistakes. Can you break this into a clear workflow without changing or retraining the model?
  ```
- **AI output summary:** Proposed separate stages for loading the model, checking labels and preprocessing, inference, saving predictions, evaluating accuracy, and analysing errors.
- **Decision:** accepted
- **Why:** Used it as a planning structure only. The notebooks implement inference and verification first, then a separate black-box confusion analysis. A plan is not evidence that the model works.
- **Problems caused:** none
---

### Understanding the Model and Preprocessing

- **Who:** RUdra
- **Goal:** Understand what `ViTImageProcessor` does and why the model expects RGB images at 224 × 224.
- **Prompt (important ones):**
  ```
  Explain what ViTImageProcessor does and why the model expects RGB images resized to 224 × 224. What should I verify before inference?
  ```
- **AI output summary:** Explained that the processor applies the expected resize, rescaling and normalization, and recommended using the processor that belongs to the selected pretrained model.
- **Decision:** accepted
- **Why:** The notebook loads `ViTImageProcessor.from_pretrained(MODEL_PATH)`, converts images to RGB and uses the recorded configuration (224 × 224, rescale 1/255, mean and std 0.5) instead of hand-picked preprocessing values.
- **Problems caused:** none
---

### Checking the 43-Class Label Mapping

- **Who:** Rudra
- **Goal:** Make sure the model's output indices match the dataset's label order.
- **Prompt (important ones):**
  ```
  The dataset has 43 traffic-sign labels and the model also has 43 outputs. Is that enough to assume the class indices match? How can I test this safely?
  ```
- **AI output summary:** Warned that matching output dimensions do not prove matching class order, and recommended comparing class names and testing predictions on a random labelled sample.
- **Decision:** accepted
- **Why:** The notebook prints both label lists and evaluates 500 shuffled clean test images (accuracy 0.942). `MODEL_TO_DATASET` stays `None` because the low-accuracy diagnostic was not triggered. Equal class counts alone were not treated as proof.
- **Problems caused:** none
---

### Building the Batch Inference Function

- **Who:** Rudra
- **Goal:** Get predicted class, confidence, true label and the full probability vector for every image across all three splits.
- **Prompt (important ones):**
  ```
  Help me design a batch inference function that returns the predicted class, confidence, true label and full probability vector for each image. It should work for the clean test split and the two corruption splits.
  ```
- **AI output summary:** Suggested `inference_mode`, batching (batch size 64), softmax over logits and argmax for the prediction, while preserving image identifiers and split names.
- **Decision:** accepted
- **Why:** The notebook defines `model_probs()` and `predict_split()` and stores split, image_id, true_label, predicted_label, confidence, correct and class_scores. I checked that confidence is the maximum softmax score and predicted_label is the argmax.
- **Problems caused:** none
---

### Handling Long CPU Inference Runs

- **Who:** Rudra
- **Goal:** Avoid losing completed work when slow CPU inference on the corrupted splits is interrupted.
- **Prompt (important ones):**
  ```
  Inference is running on CPU and the corrupted datasets take a long time. How can I avoid losing completed work if the notebook stops?
  ```
- **AI output summary:** Recommended processing in batches, saving each completed split separately, and loading existing output files when rerunning.
- **Decision:** accepted
- **Why:** The notebook checks whether each split's Parquet file exists and loads it instead of rerunning inference. The recorded run processed all 12,630 rows per split.
- **Problems caused:** none
---

### Verifying Prediction Outputs

- **Who:** Rudra
- **Goal:** Confirm the saved prediction files are complete and internally consistent.
- **Prompt (important ones):**
  ```
  What sanity checks should I run after inference to make sure the prediction files are complete and internally consistent?
  ```
- **AI output summary:** Recommended checking row counts, class-score dimensions, probability sums, confidence consistency, argmax consistency and label alignment across splits.
- **Decision:** accepted
- **Why:** The notebook checks 12,630 rows per split, 43 scores per image, probability sums close to one, confidence equal to the max score, predicted labels equal to argmax, and true-label alignment across splits. All returned OK. These checks validate consistency, not semantic correctness of each prediction.
- **Problems caused:** none
---

### Comparing Performance Across Image Conditions

- **Who:** Rudra
- **Goal:** Report the accuracy differences between clean, motion-blur and Gaussian-noise images without over-concluding.
- **Prompt (important ones):**
  ```
  The accuracy differs between clean test images, motion blur and Gaussian noise. How should I report this comparison and what should I avoid concluding?
  ```
- **AI output summary:** Suggested a compact table with sample count, accuracy and mean confidence, and distinguishing observed association from causal explanation.
- **Decision:** accepted
- **Why:** Results: clean 95.38% accuracy (mean confidence 97.76%), motion blur 92.68% (96.60%), Gaussian noise 85.00% (94.26%), each on 12,630 images. Gaussian noise gives the largest drop, but only these specific corruption conditions were tested, so I do not generalise to all real-world noise.
- **Problems caused:** none
---

### Finding the Most Common Classification Errors

- **Who:** Rudra
- **Goal:** Identify which true classes are most often confused with others while accounting for different class sizes.
- **Prompt (important ones):**
  ```
  How can I identify which true traffic-sign classes are most often confused with other classes, while accounting for different class sizes?
  ```
- **AI output summary:** Suggested grouping incorrect predictions by true and predicted label and calculating both the error count and the percentage of the true class.
- **Decision:** accepted
- **Why:** The analysis notebook groups errors by split, true_label and predicted_label, joins class sizes and computes `pct_of_class`. Raw counts alone would favour larger classes.
- **Problems caused:** none
---

### Investigating Confusion Pair 35 → 3

- **Who:** Rudra
- **Goal:** Compare how often "Ahead only" is predicted as "60 km/h" across clean, motion-blur and Gaussian-noise conditions.
- **Prompt (important ones):**
  ```
  I want to investigate why class 35 (Ahead only) is predicted as class 3 (60 km/h speed limit). Compare this confusion across clean, motion-blur and Gaussian-noise conditions.
  ```
- **AI output summary:** Recommended calculating the pair's frequency and rate in each split and inspecting examples rather than assuming a visual explanation from the labels alone.
- **Decision:** accepted
- **Why:** Results: 29/390 (7.44%) on test, 72/390 (18.46%) under motion blur, 138/390 (35.38%) under Gaussian noise. The pair becomes much more frequent under corruption, but the counts do not explain the visual cause.
- **Problems caused:** none
---

### Investigating Confusion Pair 26 → 18

- **Who:** Rudra
- **Goal:** Compare "Traffic signals" predicted as "General caution" across the same three conditions.
- **Prompt (important ones):**
  ```
  Analyse class 26 (Traffic signals) being predicted as class 18 (General caution) across the same three conditions.
  ```
- **AI output summary:** Recommended reporting the confusion rate per condition and checking correct and incorrect image examples side by side.
- **Decision:** accepted
- **Why:** Results: 54/180 (30.00%) on test, 26/180 (14.44%) under motion blur, 48/180 (26.67%) under Gaussian noise. Unlike the first pair, this one does not rise monotonically, so one pair cannot stand in for all errors.
- **Problems caused:** none
---

### Comparing Correct and Incorrect Image Properties

- **Who:** Rudra
- **Goal:** Compare brightness, contrast, image size and blue-minus-red balance between correct and incorrect predictions without overstating the result.
- **Prompt (important ones):**
  ```
  For classes 35 and 26, I calculated image brightness, contrast, image size and blue-minus-red balance. How can I compare correct and incorrect examples without overstating the result?
  ```
- **AI output summary:** Suggested summarising distributions, using medians where appropriate, and treating the statistics as descriptive rather than proof of causation.
- **Decision:** modified
- **Why:** I computed median characteristics and incorrect-minus-correct differences. For class 35 under Gaussian noise, incorrect cases have lower median brightness (62.78 vs 87.05). I describe this as an observed difference in this sample only, not as brightness causing the errors.
---

### Improving Figures and Interpretation

- **Who:** Rudra
- **Goal:** Make plots that compare confusion pairs across conditions and show which classes receive more incorrect predictions, without overlapping labels.
- **Prompt (important ones):**
  ```
  Help me make plots that compare selected confusion pairs across conditions and show which classes receive more incorrect predictions. The figures should be readable and avoid overlapping labels.
  ```
- **AI output summary:** Suggested slope charts, change bars and destination bars with consistent, readable class names.
- **Decision:** accepted
- **Why:** The notebook includes `slope_chart()`, `change_bars()` and `destinations()` with shortened class names and the confusion percentages and counts shown. I inspect each rendered figure and keep only plots that make the comparison clearer.
---

### Reviewing the Overall Conclusions

- **Who:** Rudra
- **Goal:** Separate conclusions supported by the notebook from explanations that still need testing.
- **Prompt (important ones):**
  ```
  Review these results and help me separate conclusions supported by the notebook from possible explanations that need further testing.
  ```
- **AI output summary:** Distinguished measured results (accuracy and pairwise confusion rates) from hypotheses about why corruption changes image features or predictions.
- **Decision:** accepted
- **Why:** The best-supported conclusion is lower accuracy on the corrupted splits, lowest under Gaussian noise. Confusion pairs behave differently, so no single pattern explains all errors. Limitations are stated instead of presenting AI-generated explanations as findings.
- **Problems caused:** none
---

## Limitations

- One pretrained model, one dataset and three image splits only.
- Inference ran on CPU, making full-split runs slow.
- Image statistics (brightness, contrast, minimum side, blue-minus-red) are descriptive, not a full explanation of model behaviour.
- Black-box analysis only: no retraining or attention inspection.
- Model and dataset commit hashes could not be retrieved because the connection was unavailable (recorded, not guessed).
