#!/usr/bin/env bash
# Full pipeline: dataset -> surrogates -> process window + 2-knob centring -> 4-knob centring -> dashboard
set -e
mkdir -p data models figures
python -m litho_ai.generate_dataset 8000
python -m litho_ai.train_surrogate
python -W ignore -m litho_ai.process_window
python -W ignore -m litho_ai.centring_4knob
python -m litho_ai.build_dashboard   # -> docs/recipe-console.html
# Phase 2: SEM images -> U-Net -> metrology and defect evaluation -> inspector page (~1 h on 2 CPU cores)
python -m litho_ai.sem_dataset
python -m litho_ai.sem_unet 12          # from scratch; the shipped weights were fine-tuned from an earlier version
python -W ignore -m litho_ai.sem_eval
python -W ignore -m litho_ai.sem_export
