# Cloud runs

## Description
Stuff for running in the cloud. 

## Directories
```sh
.
├── config 		# setup scripts for cloud machine
│   ├── setuproot.sh 	#
│   └── setupu0.sh      #
├── create-linode.sh    # create evo run instance
├── evo-info            # info for said instance
└── kill-linode.sh      # ./kill-linode.sh evo-info kills machine
```                       

## Notes

### Weds 17 Jan GMT 2022
- Ran experiment on `g6-dedicated-50` for about 24h (~ 34.56 GBP) on branch `doesitwork` `cd731d989d523af3e7c19ba1883692bef7c33c36` with `ga:a3a465f93482097a2ee3bc0283e3b05f9ad87fed` and `enz:671e056e7c62962e645b02d1d75f51282186deeb`

**Background:** had issues getting `bm3/main.py` working 
- `enz` - issue using `biopandas 0.2.9` - set `env.yml` to `biopandas 0.2.7` - `671e056e7c62962e645b02d1d75f51282186deeb`
- `ga` - issue getting `Pipeline` working - changed a few module io typles - `a3a465f93482097a2ee3bc0283e`
- `evo` - added `cloud/` - for provisioning machine & running etc `cd731d989d523af3e7c19ba1883692bef7c33c36`
- `evo` - modified `bm3/main.py` & `bm3/evo.sh` whilst fixing `ga` & `enz` on branch `doesitwork` & ran that on the linode `--commit-hash--`

Runtime notes:
```sh
#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate enz
for i in $(seq 8); do
	python main.py -p 128 -e 4 -n 32 -s 0.25 &
done
```
- resoure usage: 70% memory usage (~90 gb) cpu use fluctuates 2-100%
- files: first generation mostly saved ok in `bm3/runs/` - then only the scores saved to the json (which is formatted wrong). Structures were all in `/tmp` (which means that the garbage collection wasn't running or working). These were all moved to the `runs/` directory. `tar cfz runs.tar.gz runs/` & uploaded to a bucket. 
- modifications to `bm3/main.py` & `bm3/evo.sh` committed to `186161559bee0d570d00fc857dff9e2844ca0e1c`
- machine killed after about 24h

```python
np.mean(distances) - np.log(abs(affinities)) # score
```
---
### Analysis - WIP Tue Jan 18 14:38:09 GMT 2022


## Ref
```sh
┌──────────────────┬──────────────────────────────────┬───────────┬─────────┬────────┬───────┬─────────────┬──────────┬────────┬─────────┬──────┐
│ id               │ label                            │ class     │ disk    │ memory │ vcpus │ network_out │ transfer │ hourly │ monthly │ gpus │
├──────────────────┼──────────────────────────────────┼───────────┼─────────┼────────┼───────┼─────────────┼──────────┼────────┼─────────┼──────┤
│ g6-standard-16   │ Linode 64GB                      │ standard  │ 1310720 │ 65536  │ 16    │ 9000        │ 20000    │ 0.48   │ 320.0   │ 0    │
│ g6-standard-20   │ Linode 96GB                      │ standard  │ 1966080 │ 98304  │ 20    │ 10000       │ 20000    │ 0.72   │ 480.0   │ 0    │
│ g6-standard-24   │ Linode 128GB                     │ standard  │ 2621440 │ 131072 │ 24    │ 11000       │ 20000    │ 0.96   │ 640.0   │ 0    │
│ g6-standard-32   │ Linode 192GB                     │ standard  │ 3932160 │ 196608 │ 32    │ 12000       │ 20000    │ 1.44   │ 960.0   │ 0    │
│ g6-dedicated-16  │ Dedicated 32GB                   │ dedicated │ 655360  │ 32768  │ 16    │ 7000        │ 7000     │ 0.36   │ 240.0   │ 0    │
│ g6-dedicated-32  │ Dedicated 64GB                   │ dedicated │ 1310720 │ 65536  │ 32    │ 8000        │ 8000     │ 0.72   │ 480.0   │ 0    │
│ g6-dedicated-48  │ Dedicated 96GB                   │ dedicated │ 1966080 │ 98304  │ 48    │ 9000        │ 9000     │ 1.08   │ 720.0   │ 0    │
│ g6-dedicated-50  │ Dedicated 128GB                  │ dedicated │ 2560000 │ 131072 │ 50    │ 10000       │ 10000    │ 1.44   │ 960.0   │ 0    │
│ g6-dedicated-56  │ Dedicated 256GB                  │ dedicated │ 5120000 │ 262144 │ 56    │ 11000       │ 11000    │ 2.88   │ 1920.0  │ 0    │
│ g6-dedicated-64  │ Dedicated 512GB                  │ dedicated │ 7372800 │ 524288 │ 64    │ 12000       │ 12000    │ 5.76   │ 3840.0  │ 0    │
└──────────────────┴──────────────────────────────────┴───────────┴─────────┴────────┴───────┴─────────────┴──────────┴────────┴─────────┴──────┘
```
