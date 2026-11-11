# P2 AI Interaction Log - Confidence and Error Exploration

Project notebooks: `p2_confidence_exploration.ipynb`
Model / data: `kelvinandreas/vit-traffic-sign-GTSRB` on `tanganke/gtsrb` (splits: `test`, `motion_blur`, `gaussian_noise`)

---

## Summary: Failure Patterns & Best Practices

### 1. Failure Patterns to Watch For

- Unanticipated Tuple-Unpacking Errors:
  Initial AI-suggested helper functions for model loading assumed outdated return signatures from the `model_loader` module, causing runtime value errors that required explicit tuple unpacking corrections.
- Library Version Incompatibilities:
  Older Pandas and PyArrow versions installed in the local environment caused serialization warnings and silent indexing bugs when merging parquet dataframes, necessitating explicit index resets.
- Over-reliance on Mean Confidence Metrics:
  Relying solely on aggregate mean confidence values masked the dangerous reality of "silent failures"—where out-of-distribution noise drove extremely high confidence >= 0.85 on incorrect predictions.

---

### 2. Best Practices I Learned

- Enforce Parquet Caching Protocols:
  Always structure pipeline steps to check local `.parquet` artifacts before executing expensive inference or transformation loops.
- Isolate Interactive Widget State:
  When building `ipywidgets` galleries, explicitly decouple display callback functions from UI control widgets to prevent notebook freezing during rapid dropdown selections.
- Quantify High-Confidence Error Density:
  Measure the exact percentage of wrong predictions that fall into high-confidence tiers rather than just looking at overall accuracy drops.
- Cross-Verify Visual Gallery Samples:
  Always pair statistical boxplots (contrast, brightness) with interactive image inspection to ensure visual features align with numerical metrics.

---

## Log of AI Interactions

### 1. Structuring Multi-Split KDE Confidence Density Plots
- Who: Sandesh
- Goal: Build clean, publication-ready Seaborn Kernel Density Estimation plots comparing confidence distributions of correct vs. incorrect predictions across all three dataset splits.
- Prompt (important ones):
  ```
  Provide code for a 1x3 Seaborn KDE plot comparing confidence scores of correct and incorrect predictions across clean, motion_blur, and gaussian_noise splits, ensuring proper clipping between 0 and 1.
  ```
- AI output summary: Suggested using `sns.kdeplot(..., hue='correct', common_norm=False, clip=(0, 1))` across a 1x3 matplotlib subplot grid with custom color palettes.
- Decision: accepted
- Why: Produced clear visual proof of confidence degradation and shifting probability mass under noise corruptions. Automatically saved output figures to `ai_aided/figures/`.
- Problems caused: none

---

### 2. Isolating High-Confidence Misclassifications >= 0.85
- Who: Sandesh
- Goal: Programmatically extract high-confidence errors and quantify their frequency across dataset conditions.
- Prompt (important ones):
  ```
  How do we write a Pandas filtering routine to isolate predictions where correct == False and confidence >= 0.85, and compute their proportion relative to total errors across each split?
  ```
- AI output summary: Provided boolean mask slicing combined with groupby aggregations to calculate exact error ratios and mean confidence metrics for high-risk failure subsets.
- Decision: accepted
- Why: Revealed that under `gaussian_noise`, nearly half of all misclassifications maintain high confidence >= 0.85, proving the model suffers from severe out-of-distribution overconfidence.
- Problems caused: none

---

### 3. Computing Physical Image Statistics for Error Subsets
- Who: Sandesh
- Goal: Extract pixel-level metrics (brightness and contrast) for filtered error subsets and compare them against correct predictions using statistical boxplots.
- Prompt (important ones):
  ```
  Write a function to compute image brightness (mean pixel intensity) and contrast (standard deviation) for our image tensors, merge them with prediction outcomes, and generate comparative boxplots.
  ```
- AI output summary: Suggested iterating through dataset image arrays, calculating numpy-based luminance and contrast statistics, appending them to the main prediction dataframe, and plotting via Seaborn.
- Decision: accepted
- Why: Provided concrete numerical backing showing that high-confidence error images exhibit measurable drops in contrast compared to correct predictions.
- Problems caused: none

---

### 5. Building the `ipywidgets` Interactive Failure Explorer
- Who: Sandesh
- Goal: Design an interactive Jupyter widget interface with dropdown menus to filter and inspect individual traffic sign failure cases alongside top-5 class probability distributions.
- Prompt (important ones):
  ```
  Help me build an interactive ipywidgets explorer in Jupyter with dropdowns for split condition, correctness status, and confidence threshold, displaying the image alongside horizontal top-5 probability score bars.
  ```
- AI output summary: Provided `widgets.Dropdown`, `widgets.FloatSlider`, and `interactive_output` layout code using Matplotlib subplots to render the image alongside a horizontal bar chart of top-5 class scores.
- Decision: accepted
- Why: Enabled rapid qualitative auditing of model failure modes, making it easy to see *why* the model was confidently wrong (e.g., confusing speed limits with similar circular geometry).
- Problems caused: none

---

## Limitations

- Analysis focused strictly on confidence scores and physical image statistics without internal attention rollout inspection.
- Image statistics include bounding box backgrounds rather than tightly cropped sign silhouettes.
- Evaluation was limited to the clean test split and the two initial corruption splits (`motion_blur`, `gaussian_noise`).