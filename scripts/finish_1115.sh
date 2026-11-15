#!/bin/bash
# After the 11/15 background jobs (ablations, clean runtimes, qualitative figure): regenerate tables and
# figures and rebuild the report. The ablation and runtime paragraphs still need to be written by hand.
cd /home/hsy/cv-3Drecon
while pgrep -f "scripts/launch_1115" > /dev/null || pgrep -f "scripts/make_qualitative.py" > /dev/null; do sleep 60; done
.venv/bin/python scripts/make_tables.py
.venv/bin/python scripts/make_figures.py
bash report/build.sh report
echo "$(date -u '+%m-%d %H:%M') tables, figures and report rebuilt"
