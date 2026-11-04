# Traffic Sign Recognition Under Image Distortions

Minimal research repository for the JKU AI & Visualization course. The project studies
traffic-sign image classification and black-box model behavior using the
[ViT traffic-sign model](https://huggingface.co/kelvinandreas/vit-traffic-sign-GTSRB)
and the [GTSRB dataset](https://huggingface.co/datasets/tanganke/gtsrb).

The repository currently contains shared infrastructure and pair workspaces. It
does not contain research findings or completed experiments.

## Repository map

- `shared/`: neutral model, dataset, configuration, and label-verification helpers.
- `ai_aided/`: work owned by the AI-assisted pair.
- `human_only/`: work owned by the human-only pair.
- `docs/evaluation_protocol.md`: shared technical protocol and kickoff decisions.

The two pairs must keep their research code, notebooks, figures, and results in
their own directories. Only agreed technical infrastructure and evaluation
requirements belong in `shared/`.

## Setup

Python 3.11 or newer is required. The commands below work on Windows PowerShell,
macOS, and Linux with the activation command adjusted as shown.

```text
python --version
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies and check the shared package:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall shared
python -c "from shared import class_mapping, dataset_loader, model_loader; print('shared imports: OK')"
```

In VS Code, select the interpreter at `.venv\Scripts\python.exe` on Windows or
`.venv/bin/python` on macOS/Linux using **Python: Select Interpreter**.

PyTorch can use CUDA when a compatible CUDA-enabled PyTorch installation and
driver are available. This setup deliberately does not install GPU-specific
packages. The shared model loader selects CUDA only when PyTorch reports it as
available; otherwise it uses CPU.

## Moodle templates and submission

The official `team-info.json` and `workload.json` files must be copied into the
repository root when obtained from Moodle. Their schemas are not recreated
here. The midpoint checklist is:

- Public GitHub repository URL.
- Both pairs' Phase I work, `README.md`, and the official JSON templates.
- AI interaction records and `ai_reflection_midpoint.md`.
- A video link of no more than five minutes.

The video should introduce the problem, summarize both pairs' work, present
actual findings and key visualizations, and describe limitations. A suggested
outline is 0:00-0:45 problem, 0:45-2:00 pair work, 2:00-3:45 findings,
3:45-4:30 limitations, and 4:30-5:00 next steps. Midpoint deadline:
11 November 2026 at 23:59.

## Download policy

Setup and default tests do not access the network or download model weights or
datasets. Downloads occur only when code explicitly calls the model or dataset
loading helpers. Keep caches and credentials out of Git; `.gitignore` excludes
common Hugging Face caches, virtual environments, `.env`, and generated outputs.

## Project decisions still required

At the kickoff meeting, agree on the research questions, corruption conditions,
random seed, sample-matching policy, pair boundaries, and the final official
Moodle templates. See `docs/evaluation_protocol.md`.
