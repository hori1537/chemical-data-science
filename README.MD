������ LICENSE
������ Makefile           <- Makefile with commands like `make data` or `make train`
������ README.md          <- The top-level README for developers using this project.
������ data
��   ������ external       <- Data from third party sources.
��   ������ interim        <- Intermediate data that has been transformed.
��   ������ processed      <- The final, canonical data sets for modeling.
��   ������ raw            <- The original, immutable data dump.
��
������ docs               <- A default Sphinx project; see sphinx-doc.org for details
��
������ models             <- Trained and serialized models, model predictions, or model summaries
��   ������ models_ols     <- Data from third party sources.
��   ������ theme_name     <- The original, immutable data dump.
��   �@   ������ mordred     <- models fit by mordred.
��   �@   ������ deepchem    <- models fit by deepchem.
��    �@  ������ chainerchem <- models fit by chainer-chemistry.
������ notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
��                         the creator's initials, and a short `-` delimited description, e.g.
��                         `1.0-jqp-initial-data-exploration`.
��
������ references         <- Data dictionaries, manuals, and all other explanatory materials.
��
������ reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
��   ������ figures        <- Generated graphics and figures to be used in reporting
��
������ requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
��                         generated with `pip freeze > requirements.txt`
��
������ setup.py           <- Make this project pip installable with `pip install -e`
������ src                <- Source code for use in this project.
��   ��
��   ������ features       <- Scripts to turn raw data into features for modeling
��   ��   ������ build_features.py
��   ��
��   ������ models         <- Scripts to train models and then use trained models to make
��   ��   ��                 predictions
��   ��   ������ predict_model.py
��   ��   ������ train_model.py
��   ��
��   ��
��   ��
��   ������ sascorer.py     <- Scripts to run sascorer
��   ������ fpscores.pkl.gz <- pickle to run sascorer
��
������ tox.ini            <- tox file with settings for running tox; see tox.testrun.org