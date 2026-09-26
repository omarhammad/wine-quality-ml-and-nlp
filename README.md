# Wine Quality Classification & NLP Explorations

A collection of Data & AI coursework notebooks covering supervised learning on the red wine quality dataset and natural language processing with spaCy. The machine learning track compares several classifiers and ensemble methods; the NLP track explores linguistic analysis, pattern matching, similarity, and custom pipeline components.

## Project overview

| Track | Focus |
| --- | --- |
| Red wine quality | Explore physicochemical features, prepare data, select features, compare classifiers, tune SVMs, and test ensemble methods. |
| NLP with spaCy | Analyze text with tokenization, parts of speech and named entities; match patterns, compare document similarity, and extend a pipeline. |

The repository contains exploratory Jupyter notebooks, datasets, and text files. It is a set of course experiments rather than a deployed prediction service.

## Wine quality modelling

The `red_wine/` notebooks use the red wine quality data to investigate the relationship between chemical measurements and quality ratings. The workflow includes exploration and visualisation, preprocessing with imputation and scaling, feature selection with `SelectKBest`, baseline classification, and comparisons of KNN, SVM, random forest and other models. Later notebooks test voting, bagging, and stacking ensembles.

The SVM notebooks evaluate linear and RBF kernels with one-vs-one and one-vs-rest strategies, tune hyperparameters with grid search, and examine both multiclass quality ratings and a binary good/bad target. In the binary notebook, quality scores of 6 or higher are labelled good.

The notebook commentary reports that some models overfit and that tuning does not consistently improve generalisation. Its reported scores are exploratory notebook results, not a single reproducible benchmark.

## NLP with spaCy

The spaCy notebooks demonstrate:

- Tokenization, part-of-speech tags, and dependency analysis on provided texts.
- Document similarity using spaCy word vectors.
- Named entity creation and a custom component that finds the longest token.


## Repository structure

| Path | Contents |
| --- | --- |
| [`red_wine/`](red_wine/) | Wine quality datasets, exploration, preprocessing, model evaluation, and ensembles. |
| `red_wine/SVM_omar/`, `red_wine/KNN_hamed/`, `red_wine/RF_margarita/` | Focused SVM, KNN, and tree-based model experiments. |
| [`spacy_omar/`](spacy_omar/) | spaCy exercises on linguistic analysis, similarity, and a custom component. |
| [`wine_quality_dataset/`](wine_quality_dataset/) | White wine dataset supplied alongside the red wine work. |

## Technology

Python · Jupyter Notebook · pandas · NumPy · scikit-learn · imbalanced-learn · Matplotlib · Seaborn · spaCy

## Running the notebooks

1. Create a Python environment with Jupyter and the libraries used in the notebooks.
2. Install the spaCy English models required by the NLP notebooks (`en_core_web_sm` and `en_core_web_md`).
3. Open the notebook from its own folder so its relative dataset and text paths resolve.
4. For the wine workflow, the original [notebook order](red_wine/README.md) starts with exploration and preprocessing before model comparisons.

The `red_wine/requirements.txt` file is an environment export; review it before using it as an installation guide. Some notebook outputs depend on random data splits and may change when rerun. There is no single command that reproduces every experiment.
