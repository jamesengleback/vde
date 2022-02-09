#!/bin/sh
ps aux | grep app.py | awk '{print $2}' | xargs kill
