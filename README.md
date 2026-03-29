# Sample-Free Almost-Exact Estimation of Plackett-Luce Propensities for Off-Policy Ranking
This repository contains the code used for the experiments in "Sample-Free Almost-Exact Estimation of Plackett-Luce Propensities for Off-Policy Ranking" published in [ECIR 2026](https://link.springer.com/chapter/10.1007/978-3-032-21289-4_11).

Citation
----

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our [ECIR 2026 paper](https://link.springer.com/chapter/10.1007/978-3-032-21289-4_11):
```
@inproceedings{knyazev2026SampleFree,
  Author = {Knyazev, Norman and Oosterhuis, Harrie},
  Booktitle={Advances on Information Retrieval},
  Title = {Sample-Free Almost-Exact Estimation of Plackett-Luce Propensities for Off-Policy Ranking},
  Publisher = {Springer Nature Switzerland}
  Year = {2026},
  pages={163--177},
}
```

License
---

The contents of this repository are licensed under the MIT license. If you modify its contents in any way, please link back to this repository.

Usage
---
Environment
---
Create a new conda (or mamba) environment with required packages.
```
conda env create -f mpl_env.yml
conda activate mpl
```

To ensure all imports work as expected, modify the paths in the `.env` file in the project root as:
```
PROJECT_ROOT=/global/path/to/project_root
DATA_ROOT=/global/path/to/project_root/data
```

Dataset Preprocessing
---
Download the Yahoo! and MSLR datasets from their respective sources, uncompress and place the resulting folders into `./data`.

Modify paths in `./src/utils/data/datasets_info.txt` to match the absolute paths on your machine.

Generate labels for each dataset/fold. First execution may be time-consuming as raw data is first converted and stored as numpy arrays.

```
python3 mpl/utils/data/preprocessing.py
```

Policy Training (RQ1-3)
---
Both propensity estimation accuracy and sensitivity (RQ1-2), as well as Off-Policy Evaluation (OPE, RQ3) estimate marginal
propensities of Plackett-Luce policies. 
We train each dataset and policy combination using its config, specifying the fold via the command line. 
Train all models used for RQ1-RQ3 via:
```
./mpl/experiments/run_all/list_and_run.sh mpl/experiments/tmp/RQ123_models.sh mpl/experiments/run_all/RQ123_models.sh
```

Propensity Estimation (RQ1-3)
---
We then estimate propensities using either MPL or Monte-Carlo Sampling (MC). Run all consecutively via:
```
./mpl/experiments/run_all/list_and_run.sh mpl/experiments/tmp/RQ12_MPL.sh mpl/experiments/run_all/RQ12_MPL.sh  # RQ12 MPL
./mpl/experiments/run_all/list_and_run.sh mpl/experiments/tmp/RQ12_MC.sh mpl/experiments/run_all/RQ12_MC.sh  # RQ12 MC
./mpl/experiments/run_all/list_and_run.sh mpl/experiments/tmp/RQ3_propensities_MPL.sh mpl/experiments/run_all/RQ3_propensities_MPL.sh  # RQ3 MPL
./mpl/experiments/run_all/list_and_run.sh mpl/experiments/tmp/RQ3_propensities_MC.sh mpl/experiments/run_all/RQ3_propensities_MC.sh  # RQ3 MC
```

Click Generation (RQ3)
---
We further simulate user clicks for OPE, which can be obtained via: 
```
./mpl/experiments/run_all/list_and_run.sh mpl/experiments/tmp/RQ3_clicks.sh mpl/experiments/run_all/RQ3_clicks.sh
```

Evaluation and Visualization (RQ1-3)
---
Finally, the propensity (and click) estimates are processed and the results visualized via the respective jupyter notebooks:
```
jupyter lab ./mpl/experiments/RQ12.ipynb
jupyter lab ./mpl/experiments/RQ3.ipynb
```

