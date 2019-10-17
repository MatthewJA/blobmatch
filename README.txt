Blobmatch: Machine learning for cross-identification of radio surveys
James Gardner, Cheng Soon Ong, Matthew Alger
README

Abstract:
Success in radio-radio survey cross-identification is determining the real, physical objects that we're looking at. The naivest measure of two sources (or blobs) being a match for an actual object is their separation on the sky. Using this separation, we train a logistic regression classifier on the TGSS (TIFR GMRT Sky Survey Alternative Data Release 1) and NVSS (NRAO VLA Sky Survey) radio surveys. Then use its predictions to partition a patch of the sky into objects, by transitively grouping any chain of predicted matches. Although the classifier successfully learns the importance of separation, we find that the naive partitioning fails to convincingly identify objects in the sky.

Current build found at:
https://github.com/MatthewJA/blobmatch

Directory structure:
blobmatch/
	source/
		(all .ipynb notebooks, manual_labels.csv)
	report/
		pics/
			(all plots as .pdf saved by above notebooks, also cut-out comparison)
		main.tex
		report.pdf
	project/
		(non-plot outputs of notebooks except sky_matches.csv and sky_catalogue.csv, also defunct scipts and plots)
	README.txt
	LICENSE
	.gitignore

---
Guide to replicate results, please follow exactly

Blobmatch uses python 3.6.8 in jupyter notebook and has the requirements:
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

Open up a directory with the extracted surveys and all of the source code (as in source/ folder) in a jupyter notebook

Run feature_vectors.ipynb, executing all cells from top to bottom
(constructs feature vectors from source catalogues in a patch, labels based off of positional matching)
(warning: will take a few minutes to create combined catalogue, appending to a pandas dataframe is slow)
feature_vectors.ipynb requires the above TGSSADR1_7sigma_catalog.tsv and CATALOG.FIT to be present in cwd
feature_vectors.ipynb will save patch_catalogue.csv of combined match feature vectors with attached labels,
as well as tgss.csv and nvss.csv of feature vectors of individual sources

Have manual_labels.csv (found in source folder) present in cwd, manual labels made from cut-outs taken from:
http://tgssadr.strw.leidenuniv.nl/hips/
http://alasky.u-strasbg.fr/NVSS/intensity/

Run torch_logistic_regression.ipynb, executing all cells from top to bottom
(performs logistic regression using pytorch, partitions sky into physical objects)
torch_logistic_regression.ipynb requires patch_catalogue.csv (as above) and manual_labels.csv be present in cwd
torch_logistic_regression.ipynb will save weights.csv, predictions.csv, objects.csv, multi_objects.csv,
torch_lr_losses.pdf, torch_lr_weights.pdf, torch_lr_predictions.pdf, and torch_lr_partition.pdf

This ends the main-line results using logistic regression, the following are auxillary

Run sklearn_logistic_regression.ipynb, executing all cells from top to bottom
(performs logistic regression and random forest using sklearn)
sklearn_logistic_regression.ipynb requires patch_catalogue.csv and manual_labels.csv be present in cwd
sklearn_logistic_regression.ipynb will save sklearn_lr.pdf and sklearn_rf.pdf

Run score_feature_vectors.ipynb, executing all cells from top to bottom
(scores the match feature vectors in patch against various metrics, finds each individual source's best match)
score_feature_vectors.ipynb requires patch_catalogue.csv, tgss.csv, nvss.csv be present in cwd
score_feature_vectors.ipynb will save tgss_sorted.csv, nvss_sorted.csv,
hist_patch_cat_score_naive.pdf, hist_patch_cat_score_separation.pdf,
hist_patch_cat_score_spectral.pdf, and hist_patch_cat_score_combo.pdf

Run sky_positional_matching.ipynb, executing all cells from top to bottom
(constructs catalogue of primitive feature vectors over entire sky in catalogues, performs positional matching)
(warning: will take a much longer time, at least 30 minutes)
sky_positional_matching.ipynb requires TGSSADR1_7sigma_catalog.tsv and CATALOG.FIT be present in cwd
sky_positional_matching.ipynb saves sky_matches.csv, sky_catalogue.csv, hist_angle.pdf, and hist_alpha.pdf
---
