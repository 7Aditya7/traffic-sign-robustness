# Shared evaluation protocol

This document defines technical requirements and proposed choices separately.
The research questions and corruption conditions must be agreed upon by the
whole group during the kickoff meeting; none are fixed by this setup.

## Project scope and pair boundaries

The project covers traffic-sign image classification and black-box behavior
under agreed image distortions. It does not prescribe research conclusions or
empirical findings. The AI-assisted and human-only pairs may share the model,
dataset metadata, and this protocol, but each pair owns its research notebooks,
code, figures, results, and conclusions. Do not mix their implementations.

At the kickoff meeting, record the date, attendees, research question,
corruption conditions, random seed, sample-matching policy, and ownership
decisions in this document or in the relevant experiment record.

## Main research question

**To be agreed by the group:** how does the specified classifier behave when
traffic-sign images are subjected to agreed distortions?

## Primary model

The fixed primary model is `kelvinandreas/vit-traffic-sign-GTSRB`, loaded with
its matching Hugging Face image processor and image-classification model.

## Dataset and splits

The fixed dataset repository is `tanganke/gtsrb`. Inspect configurations,
features, and available splits before choosing a split. Do not assume a
corrupted test split exists or has a particular name or format.

## Input image preprocessing

Use the model's matching image processor. Inputs must be converted to RGB.
Record processor and library versions in experiment metadata.

## Class-label mapping verification

Compare dataset label IDs and names with the model's `id2label` and `label2id`.
Do not silently assume numeric IDs have identical semantics. Stop or report a
clear verification failure when mappings cannot be established.

## Inference batch size

**Proposed choice:** record an explicit batch size in every run and choose it
based on available memory. Do not silently change it between comparable runs.

## Random seed

**Proposed choice:** agree on one seed at kickoff and record it in every run.

## Prediction output schema

Each prediction record should identify the sample, split, true label ID and
name (when available), predicted label ID and name, confidence, model ID,
preprocessing metadata, condition, seed, and timestamp or run ID.

## Experimental conditions

**To be agreed by the group:** clean baseline and the exact corruption
conditions, parameter ranges, and ordering. The setup does not select them.

## Accuracy definitions

Overall accuracy is the number of correct predictions divided by the number of
evaluated samples. Report the denominator and define how unavailable labels are
handled.

## Class-specific accuracy

For a class, class-specific accuracy is correct predictions for that true class
divided by evaluated samples whose true label is that class. Report support.

## Confidence definition

Use the model's softmax probability for the predicted class, and record this
definition with outputs. Do not describe confidence as calibrated certainty.

## Pairwise clean-versus-corrupted comparisons

Compare predictions on the same sample under clean and corrupted conditions.
Report the comparison metric and denominator explicitly; do not compare
unmatched aggregate samples as if they were paired.

## Sample matching requirements

Use stable sample identifiers and preserve the same selected sample set across
conditions. Document exclusions and missing/corrupt samples.

## Reproducibility

Record code revision, Python and dependency versions, model and dataset IDs,
configuration and split, seed, processor settings, batch size, device, and
run parameters. Keep generated outputs in the owning pair directory rather
than in shared infrastructure.

## Known limitations

Model confidence may not be calibrated. Dataset metadata may expose different
configurations or splits than expected. Device, library versions, and image
decoding can affect results. These limitations must be considered when
interpreting findings.

## Kickoff decisions

- Date: `[YYYY-MM-DD]`
- Attendees: `[names]`
- Research question: `[to be agreed]`
- Corruption conditions: `[to be agreed]`
- Random seed: `[to be agreed]`
- Other decisions: `[to be recorded]`
