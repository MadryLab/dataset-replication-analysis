# Identifying Statistical Bias in Dataset Replication
Code for the paper "Identifying Statistical Bias in Dataset Replication." The original paper can be found on [here](http://gradientscience.org/data_rep_bias.pdf) ([Blog post](http://gradientscience.org/data_rep_bias/)) .

## Annotation Data
Annotation data not included in the Git repository is necessary to run any
analyses (approx 2 GB). The annotation data can be found
[here](https://www.dropbox.com/sh/6qtf2iqqbuc4r30/AACeMh0mUleGG_JhOkvAgYBra?dl=0).
There are two Pandas dataframes saved with pytorch (e.g. load them with
`torch.load(path)`) that correspond to both the raw annotation data and the data
cleaned version (data cleaning described in paper). The is also a dataframe
containing raw data from the original ImageNet-v2 study, which is required by
``run_orig_data.py``. 

## Beta-Binomial fitting
To adjust for accuracy using parametric modelling (cf. Section 5.3), run `run_betabinom.py` while inputting the path of a dataframe retrieved from the "Annotation Data" section above to `--df-path`.

```
usage: run_betabinom.py [-h] --out-dir OUT_DIR --df-path DF_PATH [--debug]

optional arguments:
  -h, --help         show this help message and exit
  --out-dir OUT_DIR  Out directory to save results to
  --df-path DF_PATH  Input dataframe to draw annotations from
  --debug
```
For example:

```
python run_betabinom.py --df-path df_clean.pt --out-dir output/
```

After this program runs, to visualize/parse the results you will need to open up the iPython notebook `src/Beta-Binomial Model Analysis.ipynb` and replace the `INPUT_DATA` variable as instructed by the printout at the end of `run_betabinom.py` before running all the cells. The notebook will output both a visualization as well as a Pandas dataframe with accuracies and adjusted accuracies for each model.

## Jackknife fitting

Similarly, ``run_jackknife.py`` will run the statistical jackknife correction
(cf. Section 5.2) using the dataframe retrieved from the "Annotation Data"
section above.

```
usage: run_jackknife.py [-h] --out-dir OUT_DIR --df-path DF_PATH [--debug]

optional arguments:
  -h, --help          show this help message and exit
  --out-dir OUT_DIR   Out directory to save results to
  --df-path DF_PATH   Input dataframe to draw annotations from
  --delete-d D        Runs the delete-D jackknife (default 1)
  --num-replicates N  Number of replicates to use to estimate expectations (default 100)
  --workers W         Number of threads to use (default 2)
  --save-justification    Also output plot of bias vs 1/n (cf. Figure 12)
```
For example:

```
python run_jackknife.py --df-path df_clean.pt --out-dir output/ --save-justification
```

## Original data analysis
Finally, to run the analysis in Appendix D (original data analysis), run
``analyze_orig_data.py`` using the dataframe dowloaded in the first section.

```
usage: analyze_orig_data.py [-h] --out-dir OUT_DIR --df-path DF_PATH --experiment {naiveest, heldout, ezflickr}

optional arguments:
  -h, --help         show this help message and exit
  --out-dir OUT_DIR  Out directory to save results to
  --df-path DF_PATH  Input dataframe to draw annotations from
  --experiment       Which Appendix D experiment to run (D.1, D.2, D.3)
  --workers          Number of threads to use (default 2)
  --trials           Number of trials to average over (default 10)
```
