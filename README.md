# Densifier - (Orthogonal) Transformation for Word Embeddings

An implementation of the _Densifier_ introduced by _Rothe et al. 2016_ which aims at grouping words based on any given separating signals such as sentiment, concreteness, or frequency, as long as embeddings encode them.

### General Idea and Training Objective

The training objective is to group words in an ultradense space, e.g. dim=1, according to provided separating signals. Setting the ultradense space with dim==1 yields lexicons based on embeddings.

### Further Information about the Codes

These codes are optimized such that they **only** work when dim==1. However, it should be straightforward enough to modify them to output ultradense spaces with dim>1, which can be subsequently feed to NNs.

These codes are written in **NumPy**. For implementations using autograd frameworks, one can refer to [here](https://github.com/JULIELab/wordEmotions) (for TensorFlow users) and [here](https://github.com/williamleif/socialsent) (for Keras users). Note, running _Densifier_ on GPU may not be ideal -- there are some overheads moving around from tensor to ndarrays (also GPU <-> CPU) for doing the expensive SVD, see this [thread](https://github.com/tensorflow/tensorflow/issues/13222).

### Requirements and Compatibility

- Python 2.7
- NumPy 1.14.3
- SciPy 1.1.0

All codes are written in Python 2.7, yet should be compatible with Python 3.

### Usage

```
python Densifier.py
--LR		learning rate
--alpha		hyperparameter balancing two sub-objectives
--EPC 		epochs
--OUT_DIM	ultradense dimension size
--BATCH_SIZE	batch size
--EMB_SPACE	input embedding space
--SAVE_EVERY	save every N steps
--SAVE_TO	output trained transformation matrix
```

### Results

Using the same Twitter embedding space in  _Rothe et al. 2016_, these codes perform roughly the same to the  [TensorFlow implementation](https://github.com/JULIELab/wordEmotions) --  ~~0.48 v.s. 0.47 of kendall's tau on the SemEval2015 10B sentiment analysis task, which are unfortunately both lower than 0.65 reported by the original author~~ 0.63 which are close to the reported value 0.65. Note, need to carry out evaluations on the joint vocabulary.

To replicate results in other languages and domains, the original test sets are required but I did not find them.

### Some Discussion

- **Efficiency**:  _Rothe el al. 2016_ reported that all experiments were finished in 5 mins. Unfortunately I am not able to achieve this speed as it takes me ~0.2s to compute an SVD of a 400 x 400 dense matrix using NumPy. Autograd frameworks can take more time.
- **Orthogonal Constraint**: _Buechel et al. 2018_ reports that enforcing the orthogonal constraint introduces no difference on performance. Similar observations occur in this implementation. Releasing the orthogonal constraint means we now can touch the norm of $Q$ so probably we need to regularize the $l2$ norm of $PQ$ such that the loss does not go to $-\infty$. Again, no improvements are observed in this case either. The orthorgonal constraint regularizes optimization steps go along the surface of the cube, but probably is not significantly helpful when the evaluation metric is ranking based. 

### References

Papers referred in this implementation:

```
@InProceedings{N16-1091,
  author = 	"Rothe, Sascha
		and Ebert, Sebastian
		and Sch{\"u}tze, Hinrich",
  title = 	"Ultradense Word Embeddings by Orthogonal Transformation",
  booktitle = 	"Proceedings of the 2016 Conference of the North American Chapter of the      Association for Computational Linguistics: Human Language Technologies    ",
  year = 	"2016",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"767--777",
  location = 	"San Diego, California",
  doi = 	"10.18653/v1/N16-1091",
  url = 	"http://aclweb.org/anthology/N16-1091"
}

@InProceedings{N18-1173,
  author = 	"Buechel, Sven
		and Hahn, Udo",
  title = 	"Word Emotion Induction for Multiple Languages as a Deep Multi-Task      Learning Problem    ",
  booktitle = 	"Proceedings of the 2018 Conference of the North American Chapter of the      Association for Computational Linguistics: Human Language Technologies,      Volume 1 (Long Papers)    ",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1907--1918",
  location = 	"New Orleans, Louisiana",
  doi = 	"10.18653/v1/N18-1173",
  url = 	"http://aclweb.org/anthology/N18-1173"
}

```

