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

The script `process_SNSPD.py` imports and processes all the data and draws Fig. 4.

## SPADs

Contains all measured counts for the SPADs. Each text file is in the same format as the counts for the SNSPD, except there are 30 repetitions and 20 s measurement time.

The script `process_SPADs.py` imports and processes all the data. Fitting both paralyzing and nonparalyzing models is included.

The Jupyter notebook `model_corrections.ipynb` provides the analysis of the counts corrected for dead time and dark counts. Fig. 11 is reproduced from scratch with all technical steps explicitly given in the form of code and notes.

The Jupyter notebook `afterpulses_difference.ipynb` shows how much the nonlinearity would change if the full temporal distribution of afterpulses was taken into account. The detector taken as an example is SPAD-1 with its afterpulsing distribution provided in the `afterpulsing` folder \[1,2\]. Fig. 13 is drawn.

The Jupyter notebook `efficiency_settling_hypothesis.ipynb` explores the hypothesis that the supralinearity of SPAD-3 is caused by an efficiency settling effect. The necessary magnitude of the effect is shown and then compared to time-resolved measurements of SPAD-3. The slope of the interarrival time statistics corresponds to a constant efficiency and rules out the hypothetical effect. Figs. 15 and 16 are provided.

The Jupyter notebook `ad_hoc_hypothesis.ipynb` explores a hypothetical empirical model that would match the nonlinearity data of SPAD-3. The extra polynomial term is then attributed to dark counts and efficiency to show that no such phenomena are plausible. Fig. 14 is provided.

## Stability

Both text files contain a series of intensities measured during a 12-hour stability measurement. The file `stability_SLED.txt` represents the source alone, while `stability_SLED_NL.txt` includes the whole measurement setup. The script `Allan_deviation.py` computes the Allan deviations and draws the plots in Fig. 6.

## References

* \[1\] I. Straka, J. Grygar, J. Hloušek and M. Ježek, *Counting Statistics of Actively Quenched SPADs Under Continuous Illumination*, Journal of Lightwave Technology 38, 4765 - 4771 (2020). https://doi.org/dzmt
* \[2\] I. Straka, J. Grygar, J. Hloušek and M. Ježek, https://doi.org/10.24433/CO.8487128.v1, *Counting Statistics of Actively Quenched SPADs Under Continuous Illumination*, CodeOcean capsule (2020).
