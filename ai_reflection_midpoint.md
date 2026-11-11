# AI Reflection at Midpoint

## Names and student IDs

- Student 1: `Rudra Prasad Dhungana — K12534884`
- Student 2: `SANDESH GHIMIRE — K12452503`

## Context and objectives

Our project is "Beyond Accuracy: Investigating Vision Transformer Behavior in Traffic Sign Recognition". The group has four members in two pairs. Pair A (P1 and P2) works with generative AI (Claude); Pair B works without it. Both pairs investigate the same model and data:

- Model: `kelvinandreas/vit-traffic-sign-GTSRB`, a Vision Transformer fine-tuned from `google/vit-base-patch16-224-in21k`, with 43 traffic sign classes. We did not train or fine-tune anything.
- Data: `tanganke/gtsrb`, using the `test`, `motion_blur` and `gaussian_noise` splits (12,630 images each, 37,890 predictions in total).

Phase I asks "How does the model behave?", treating the model as a black box. Our objective is to go beyond accuracy and find where the model errs or becomes uncertain, and how image corruptions change this. In Pair A, P1 covers corruption and confusion analysis and P2 covers confidence and the image explorer. This report covers P1's work in detail.

Phase II (P2)2: P2 covers model confidence distributions, error tracking, identifying high-confidence incorrect predictions, physical image metric statistics, and an interactive `ipywidgets` case explorer allowing detailed inspection of individual predictions, ground truths, and top-5 probability distributions.

## How GenAI was used

P1 used Claude in a chat interface for the following. Preserved interactions and their outcomes are in `ai-interactions/p1_interactions.md`.

- **Set-up and inference (entries 1, 2, 7).** Step-by-step guidance and an inference notebook (`common/inference.ipynb`) that loads the model, checks class labels, tests accuracy on a random sample, runs all three splits, verifies the output, and records software versions. It produced `predictions.parquet` with split, image_id, true_label, predicted_label, confidence, correct and class_scores.
- **Debugging (entries 3 to 6, 12).** Diagnosing a missing package, a Brotli decompression error hidden behind a misleading "can't load configuration" message, very slow downloads (we switched to manual downloads and local loading), and a failed split-loading attempt. Also progress and timing feedback for slow CPU inference.
- **Analysis code (entry 8).** A notebook with class error rates (with class sizes), confusion pairs, the same pair across conditions, an over-prediction ("sink") table, image galleries, and image statistics.
- **Analysis ideas (entry 10).** Choosing one pattern for deep investigation (Ahead only predicted as 60 km/h), and the idea of comparing the same image_id across conditions.
- **Visualization code (entry 11).** A slope chart, change bars, destination bars and an interactive explorer, plus a later fix for readability problems.
- **Communication.** Drafts of a findings document, presentation slides (on the course template) and a spoken script for the video. `[Add these to the AI log if not yet there.]`

The final model and dataset were chosen by our group plan. Earlier, AI had suggested other model and dataset candidates, which we did not use.

Phase II: The aim was to show confusion patterns, confidence behavior, and how they change under corruption, not only an accuracy number. Alternatives and decisions:

- Matrix vs ranked views: A full 43×43 confusion matrix was discussed, but it is dominated by the diagonal and hard to read at that size. Our figures use ranked pairs and bars instead, always with counts next to percentages.
- Slope chart & Distribution density plots: KDE density plots comparing correct versus incorrect confidence across clean and corrupted conditions clearly illustrate confidence collapse.
- Diverging change bars & Destination bars: for corruption shifts and error trajectories.
- Image-based views: galleries of wrong and correct examples, showing original images across conditions and boxplots comparing brightness, contrast, and color balance of correct versus wrong images.
- Interactive explorer: with dropdowns for condition and error filtering, displaying real traffic sign images alongside horizontal top-5 probability score bars. Built with `ipywidgets` directly inside the Jupyter notebook.

After checking the AI-written charts on the real data, we refined the design for maximum readability.

## Visualization design collaboration

The aim was to show confusion patterns and how they change under corruption, not only an accuracy number. Alternatives and decisions:

- **Matrix vs ranked views.** A full 43×43 confusion matrix was discussed, but it is dominated by the diagonal and hard to read at that size. Our figures use ranked pairs and bars instead, always with counts next to percentages.
- **Slope chart** of selected confusions across clean, blur and noise. It shows that confusions react differently: 35 → 3 grows, 38 → 17 falls under noise, and 26 → 18 halves under blur.
- **Diverging change bars** (corrupted minus clean, in number of images) for each corruption, so we see what the corruption creates or removes.
- **Destination bars** for one true class (Ahead only), showing where its images go in each condition.
- **Image-based views:** galleries of wrong and correct examples, one image shown across the three conditions, and boxplots comparing size, brightness, contrast and colour balance of correct and wrong images.
- **Interactive explorer** with dropdowns for condition, true class and predicted class, showing the real images. It is built with ipywidgets in the notebook. `[Confirm the status of the Streamlit dashboard from the task division: built, partly built, or not built.]`

After checking the AI-written charts on the real data, we changed the design (see the next section).

## Criticism and refinement

- **Readability on real data.** The charts the AI wrote worked on its synthetic test data but, on our data, legend labels were cut mid-word, value labels overlapped at the clean condition, and the shared axis in the destination chart hid small bars. We replaced class names with short readable names, de-overlapped labels, moved the legend below the plot, and used the same classes on every panel.
- **Interpretation.** From the galleries the AI described the wrong Ahead only images as "dark, backlit". The image statistics corrected this: wrong images are on average brighter (median 101 vs 53), higher in contrast (56 vs 26) and smaller (36 vs 42 px side), which fits a dark sign against a bright background. Colour balance showed no consistent difference, so we dropped that hypothesis.
- **A wrong class size.** The AI stated that class 26 has 360 test images. Our notebook printed 180. We now print n next to every rate.
- **Our own extensions.** P1 extended the notebook with a side-by-side view of one image across all three conditions and with image statistics for classes 35 and 26 in every condition. `[Confirm which of these cells were written by P1 and which with AI help.]`

## Investigation direction

Findings so far (all from actual predictions, details in `results/p1_findings.md`):

- Accuracy is 95.4% (clean), 92.7% (blur) and 85.0% (noise); mean confidence falls less (97.8%, 96.6%, 94.3%).
- Ahead only is predicted as 60 km/h for 29, 72 and 138 of 390 images (7.4%, 18.5%, 35.4%). Under noise this wrong class is more frequent (138) than the correct one (126).
- Comparing the same images, all 91 images wrong on clean data stay wrong under noise, and 173 of the 299 correct ones (58%) become wrong.
- Noise: the largest increases all end in 60 km/h, 80 km/h or Roundabout (+109 for Ahead only → 60 km/h, +74 for Priority road → Roundabout). Blur mostly adds confusions between related signs, plus Ahead only → 60 km/h (+43).
- Not every confusion grows: Keep right → No entry falls under noise (44 → 26), and blur rescues 34 of 67 wrong Traffic signals images while breaking 6 of 113 correct ones.
- Wrong images tend to be smaller, brighter and higher in contrast, but the groups overlap a lot and no significance test was run.

Our results are descriptions of behavior, not explanations. Direction for Phase II: ask why the model maps Ahead only to 60 km/h, compare saliency or attention and embeddings for correct and wrong images, and test whether noise sends many classes to the same few outputs.

Phase II: For this, we have to investigate how corruption impacts model calibration and uncertainty, quantify the proportion of high-confidence errors >= 0.85$ and >= 0.95 under perturbation, and test whether physical image properties (such as contrast) correlate with the model's tendency to fail silently while remaining overconfident.

## Verification of AI suggestions

- **Set-up checks:** class names compared between model and dataset; accuracy on a random sample of 500 clean images (94.2%) before any analysis, then on the full split (95.4%).
- **Automatic checks on the predictions:** 12,630 rows per split, labels aligned across splits by image_id, 43 scores summing to 1, confidence equal to the maximum score, predicted label equal to the argmax.
- **Counts next to every rate**, and every number in the findings taken from the notebook output.
- **Running AI code on the real data** and opening every figure; this found the readability problems above.
- **Checking interpretations with numbers**, which corrected the "dark" description.
- **Our own measurements instead of the model card:** the card reports 98.46% without naming the data; we measured 95.4% on our test split and report only that.

Limitations: the AI tested its code on synthetic data, which finds crashes but not readability or real-data behavior. The image statistics include the background, not only the sign. Galleries show random samples of 10 images. Only two classes were studied in depth, and the model's training data is unknown.

P2: Cross-referencing confidence score distributions against correctness flags, verifying high-confidence error counts 0.85 and 0.95 against aggregated split summary tables, and inspecting individual sample predictions through the interactive ipywidgets gallery to ensure alignment between numerical outputs and visual data.

## Comparison with the human-only pair

Observations from Pair A's side:

- **Speed and breadth.** AI helped us produce many views quickly (tables, four chart types, an interactive explorer) and suggested the same-image comparison across conditions, which produced our strongest result. A lot of time, however, went into environment problems (packages, downloads, slow CPU inference of about 44 minutes per split) and into checking AI-written code.
- **Cost of verification.** Several AI suggestions needed correction (see Failure patterns), so confidence in a finding came from our own counts and statistics, not from the AI's wording.

- Insights & Convergence: Both pairs converged on identical core findings regarding noise vulnerability and overconfidence.

- Visualization & Breadth: AI accelerated multi-variable exploratory views and interactive widgets, while the human-only pair produced manual static figures.

- Confidence & Pipeline: Both teams share high confidence based on matching metrics; Pair B used independent scripting rather than the shared pipeline.

## Failure patterns

- **Untested fixes:** a suggested change to load the corrupted splits failed with `Unknown split "motion_blur"`, and an earlier version still downloaded data from the internet although local files were set up (entries 4 and 5).
- **Facts that were remembered or inferred, not checked:** the class-26 size (360 instead of 180), and model and dataset details given from memory.
- **Plausible but wrong interpretation:** reading the errors as "dark" when the statistics showed brighter images (entry 10).
- **Tests that missed real problems:** synthetic data did not reveal unreadable chart labels (entry 11). The AI's own test did catch a plotting option that newer matplotlib versions no longer support, before delivery (entry 8).
- **Guessing from partial information:** the AI needed the full error traceback to find the real cause of a loading error, and its first guess about a stalled progress bar was only resolved by printed timing (entries 3 and 6).

Phase II: 
Untested dataset split configurations: AI-suggested loading methods threw ConnectionResetError exceptions when attempting remote fetches, requiring manual redirection to local parquet caches.

Dependency version mismatches: AI-generated code snippets occasionally referenced incompatible legacy arguments across PyArrow, NumPy, and Pandas versions, requiring manual correction during local notebook execution.

## Lessons learned

- Give the AI the full error message or output, not only the last line.
- Run on a small sample first, and keep the sanity checks inside the notebook.
- Print the number of images (n) next to every rate.
- Treat AI hypotheses as hypotheses: test them with numbers, and report the ones the data does not support (the colour hypothesis here).
- Open every generated figure on the real data before using it.
- AI saved effort on code and broadened the analysis, but it did not remove the verification work, and the results we trust are the ones we checked ourselves.
- Log AI mistakes immediately, while it is still clear what happened.
- Cache dataset splits and predictions locally to prevent connection drops during interactive exploration.
- Verify confidence thresholds empirically, as out-of-distribution models frequently exhibit extreme overconfidence on incorrect predictions.
