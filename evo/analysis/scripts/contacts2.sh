#!/bin/bash
find ../runs/newscore -type d -links 2 | parallel python docking-analysis-2.py > docking-analysis2.1.json & disown
