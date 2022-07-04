#!/bin/bash

#linode-cli obj get james-engleback evo-a-runs.tar.gz
tar xfz evo-a-runs.tar.gz &
linode-cli obj get james-engleback evob-runs.tar.gz
tar xfz evob-runs.tar.gz
