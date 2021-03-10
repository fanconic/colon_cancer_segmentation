# Cancer Segmentionion - ML4H Project1
Project 1 one the course Machine Learning for Health Care at ETH Zürich

## How to run on the Leonhard Cluster:
`bsub -n 4 -W HH:MM -N -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" ./run.sh`

Authors:
- Claudio Fanconi
- Manuel Studer

