#!/bin/bash
uwsgi -s /tmp/vde.sock --manage-script-name --mount /vde=server:app

source ~/miniconda3/etc/profile.d/conda.sh
conda activate enz
export FLASK_APP=app.py
python app.py 5001
