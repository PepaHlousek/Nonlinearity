# Experimental observation of anomalous supralinear response of single-photon detectors
Josef Hloušek, Ivo Straka, Miroslav Ježek (Palacký University Olomouc)

[![preprint](https://img.shields.io/badge/arXiv-2109.08347-b31b1b.svg)](https://arxiv.org/abs/2109.08347)

This repository provides the data for the manuscript *Experimental observation of anomalous supralinear response of single-photon detectors* available as preprint on [arXiv:2109.08347](https://arxiv.org/abs/2109.08347)

Tested with Python >= 3.8.13, matplotlib 3.5.1, numpy 1.21.5, scipy 1.7.3, re 2.2.1, sympy 1.10.1, lmfit 1.0.3.

## SNSPD

Contains all data relevant to the SNSPD. The bias current values are contained in the file names. Each data file consists of one entry per line. 

| File              | Entry format                | Note                                       |
|-------------------|-----------------------------|--------------------------------------------|
| SNSPD_DC.txt*     | [dark count rate] [+-error] |                                            |
| SNSPD_eff.txt*    | [efficiency] [+-error]      | measured relative to spec for 25 microAmps |
| SNSPD_effplot.txt | [bias current] [efficiency] | manufacturer's specs                       |
| deadtime.dat*     | [dead time] [+-error]       |                                            |
| i[XXX].txt        | [A] [B] [AB]                | counts; 20 repetitions, 30 s meas. time    |

\* each entry corresponds to a bias current XXX

The script `process_SNSPD.py` imports and processes all the data and draws Fig. 4 of the main text.

## SPADs

Contains all measured counts for the SPADs. Each text file is in the same format as the counts for the SNSPD, except there are 30 repetitions and 20 s measurement time.

The script `process_SPADs.py` imports and processes all the data. Fitting both paralyzing and nonparalyzing models is included.

The Jupyter notebook `model_corrections.ipynb` provides the analysis of the counts corrected for dead time and dark counts. Fig. 7 from the supplemental material is reproduced from scratch with all technical steps explicitly given in the form of code and notes.

## Stability

Both text files contain a series of intensities measured during a 12-hour stability measurement. The file `stability_SLED.txt` represents the source alone, while `stability_SLED_NL.txt` includes the whole measurement setup. The script `Allan_deviation.py` computes the Allan deviations and draws the plots in Fig. 2 of the supplemental material.
