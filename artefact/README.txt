Cross identification of radio astronomy objects using machine learning
James Gardner, Cheng Soon Ong, Matthew Alger
README

This project considers the problem of cross identification of objects in radio astronomy. Cross identification is the task of matching objects in one sky survey to the corresponding object in a different sky survey. We implement positional matching and machine learning based binary classification for solving the task of cross id. The methods are applied to NVSS (NRAO VLA Sky Survey) and TGSS (TIFR GMRT Sky Survey Alternative Data Release 1).

Cut-outs taken from:
http://tgssadr.strw.leidenuniv.nl/hips/
http://alasky.u-strasbg.fr/NVSS/intensity/

positionalmatching.ipynb takes two lists of celestial co-ordinates and creates a mapping for each into the other, taking the nearest point within a accepted distance.



---
Guide to replicate results, please follow exactly

Blobmatch uses python 3.6.8 in jupyter notebook and has the following requirements:
astropy==3.2.1
ipython==5.5.0
jupyter==1.0.0
jupyter-client==5.2.2
jupyter-console==6.0.0
jupyter-core==4.4.0
matplotlib==3.0.3
numpy==1.16.2
pandas==0.24.2
scikit-learn==0.21.3
sklearn==0.0
torch==1.1.0
torchvision==0.3.0
tqdm==4.33.0

Download the TGSS and NVSS radio object surveys from the links below and unzip them
https://github.com/MatthewJA/blobmatch/releases/download/v0.1/TGSSADR1_7sigma_catalog.tsv.gz
https://github.com/MatthewJA/blobmatch/releases/download/v0.1/CATALOG.FIT.gz

Run feature_vectors.ipynb, executing all cells from top to bottom
feature_vectors.ipynb requires the above two files TGSSADR1_7sigma_catalog.tsv and CATALOG.FIT to be present in cwd
feature_vectors.ipynb will produce patch_catalogue.csv of feature vectors with attached labels

Run logistic_regression.ipynb, executing all cells from top to bottom
logistic_regression.ipynb requires the above patch_catalogue.csv be present in cwd
logistic_regression.ipynb will produce weights.csv, predictions.csv, objects.csv, multi_objects.csv, along with torch_lr_losses.pdf, torch_lr.pdf, and torch_lr_predictions.pdf

---

