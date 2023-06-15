# Methods_ecol_evol_2023
This code is an implementation of the framework described in the paper: A framework for identifying factors controlling cyanobacterium Microcystis flos-aquae blooms by coupled CCM-ECCM Bayesian networks. Authors: Ofir Tal, Ilia Ostrovsky and Gideon Gal.

# Instructions


# The code is implemented in Python 3.9 (Linux OS, Anaconda) using the following packages:
auto_shap==0.1.1
bnlearn==0.7.8
gseapy==0.10.7
matplotlib==3.3.2
networkx==3.0
numpy==1.21.5
pandas==1.4.3
pydot==1.4.2
pyEDM==1.14.0.2
pymongo==3.11.4
scikit_learn==1.2.2
scipy==1.8.0
seaborn==0.11.2
shap==0.41.0
skccm==0.2.dev0
tqdm==4.62.3

# Output files:





# Run - recommended approach

mkdir xxx

cd xxx

virtualenv xxx

source xxx/bin/activate

pip install -r requirements.txt

Execute by:

python 1_CCM1-CCM2_ECCM.py
 - Here examine ECCM results, curated output is CCM_ECCM_curated.csv 
python 2_SURR.py
python 3_DAG-BN.py


