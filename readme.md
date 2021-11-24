# `evo` - virtual directed evolution

## Overview
Virtual directed evolution program to rengineer P450 BM3 to bind to the herbicide mesotrione such that the target carbon for hydroxylation (carbon 5) is close to the heme.

Uses a genetic algorithm (from `ga`) to generate pools of mutants, whose structures are predicted using `pyrosetta` and mesotrione docked usign `vina` (all wrapped in `enz`). 
Score function is defined in `evo/score.py`

## Layout
```bash
.
├── analysis
│   ├── analysis_old
│   ├── models
│   ├── outputs
│   └── scripts
├── data
│   ├── 4KEY.pdb
│   ├── HRAC_Herbicides.csv
│   └── mesotrione.png
└── evo
    ├── __pycache__
    ├── bm3.py
    ├── evo.py
    ├── evo.sh
    ├── old_runs.tar.gz
    ├── runs
    ├── score.py
    ├── test_run
    ├── wt
    └── wt.py
```
