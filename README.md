# SpEC
Sparsity, Explainability, and Communication. 

This is the repository for [The Explanation Game: Towards Prediction Explainability through Sparse Communication](https://www.aclweb.org/anthology/2020.blackboxnlp-1.10.pdf), accepted at BlackBoxNLP 2020. 


### Usage

First, create a virtualenv using your favorite tool, and then install all dependencies using:

```bash
pip3 install -r requirements.txt
```

Finally, install it as a library with:
```bash
python3 setup.py install
```

You can see a complete help message by running:
```bash
python3 -m spec --help
```


### Data

1. Download the datasets with the script `download_datasets.sh` (1.6G). 
Yelp dataset should be downloaded separately (6G).

2. Then run the script `bash generate_dataset_partitions.sh` to create train/dev/test partitions for AgNews, IMDB and Yelp.

3. If you want to use GloVe embeddings (as in our paper), you have two options:

    a) Use the script `download_glove_embeddings.sh` to download all embedding vectors. 
And, if you want to use only the embeddings for a particular corpus, i.e.,
restrict the embeddings vocabulary to the corpus vocabulary for all downloaded corpus,
use the script `scripts/reduce_embeddings_model_for_all_corpus.sh`. 

    b) Download the already restricted-to-vocab glove embeddings with the script `download_restricted_glove_embeddings.sh`.

Here is how your data folder will be organized:
```
data/
├── corpus
│   ├── agnews
│   ├── imdb
│   ├── snli
│   └── sst
│   └── yelp
├── embs
│   └── glove
```


### How to run

Basic usage:
```bash
python3 -m spec {train,predict,communicate} [OPTIONS]
```

You can use the command `train` to train a classifier. 
Alternatively, if you are interested only in the communication part,
you can download trained classifiers here:
- [softmax-based models](https://www.mediafire.com/file/vqqiabataunkczs/softmax-models.zip/file) (743M)
- [entmax-based models](https://www.mediafire.com/file/eb994fafkjx2fvk/entmax15-models.zip/file) (743M)
- [sparsemax-based models](https://www.mediafire.com/file/jjczy5dvp7nt6ih/sparsemax-models.zip/file) (743M)

After downloading and extracting, put them in `data/saved-models/`. 
Or download all of them with the script `download_trained_models.sh`. 
  
For training the classifier-explainer-layperson setup, you can use the command `communicate` 
and set the path to load the trained classifier via the `--load` argument. For example, see [experiments/train_sst.sh](https://github.com/deep-spin/spec/blob/master/experiments/train_sst.sh) and [experiments/communicate_sst.sh](https://github.com/deep-spin/spec/blob/master/experiments/communicate_sst.sh).

Statistics will be displayed during training. Here is a snippet of what you'll see:
```
Loss    (val / epoch) | Prec.     Rec.    F1     (val / epoch) | ACC    (val / epoch) | MCC    (val / epoch) | TVD    (val / epoch) | ACC L  | ACC C  | 
----------------------+----------------------------------------+----------------------+----------------------+----------------------+--------+--------+
 0.6495 (0.6495 /  1) | 0.8005   0.7562   0.7482 (0.7482 /  1) | 0.7579 (0.7579 /  1) | 0.5550 (0.5550 /  1) | 0.4105 (0.4105 /  1) | 0.7350 | 0.9504 |
```
The accuracy of the communication (ACC) represents the CSR proposed in our paper, ACC L is the accuracy of the layperson model, and ACC C is the accuracy of the classifier.

Take a look in the `experiments` folder for more examples. 



### Human annotations

(we'll provide a download link soon!)

For now, enter in contact if you want access to 200 hundred annotated examples of IMDB and SNLI for each
explainer in Table 4. Check the supplemental material in our paper for more information about the annotation process.  


### MT experiments

Machine Translation experiments were carried in [possum-nmt](https://github.com/deep-spin/possum-nmt), 
DeepSPIN's private version of [joey-nmt](https://github.com/bastings/joey-nmt) created
by Ben Peters. 
Here is the config file that I used to train and save (manually) gradients and attention probabilities: 
[config_iwslt17_sparsemax.yaml](https://gist.github.com/mtreviso/f95f4498c7f71079b3e5d07840c2bc89). 
Let me know if you want access to this data (+20GB).


### Citation
```
@inproceedings{treviso-martins-2020-explanation,
    title = "The Explanation Game: Towards Prediction Explainability through Sparse Communication",
    author = "Treviso, Marcos  and
      Martins, Andr{\'e} F. T.",
    booktitle = "Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.blackboxnlp-1.10",
    pages = "107--118",
}
```


### License

MIT.
