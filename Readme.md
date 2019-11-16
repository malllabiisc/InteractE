## InteractE: Improving Convolution-based Knowledge Graph Embeddings by Increasing Feature Interactions

[![Conference](http://img.shields.io/badge/AAAI-2020-4b44ce.svg)](https://aaai.org/Conferences/AAAI-20/) [![Paper](http://img.shields.io/badge/paper-arxiv.1911.00219-B31B1B.svg)](https://arxiv.org/abs/1911.00219)

Source code for [AAAI 2020](https://aaai.org/Conferences/AAAI-20/) paper: **InteractE: Improving Convolution-based Knowledge Graph Embeddings by Increasing Feature Interactions**

![test image size](/Users/shikharvashishth/server/scratchd/home/shikhar/rel_gcn/InteractE/overview.png){:height="700px" width="400px"} **Overview of InteractE:** *Given entity and relation embeddings, InteractE generates multiple permutations of these embeddings and reshapes them using a "Chequered" reshaping function. Depthwise circular convolution is employed to convolve each of the reshaped permutations, which are then fed to a fully-connected layer to compute scores. Please refer to Section 6 of the paper for details.*

### Dependencies

- Compatible with PyTorch 1.0 and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use FB15k-237,  WN18RR and YAGO3-10 datasets for evaluation. 
- FB15k-237,  WN18RR are included in the repo. YAGO3-10 can be downloaded from [here](https://drive.google.com/drive/folders/186yl5MetAx_ialN0fOCvrtnBiKRJdzXO?usp=sharing). 

### Training model from scratch:

- Install all the requirements from `requirements.txt.`

- Execute `sh preprocess.sh` for extracting the datasets and setting up the environment. 

- To start training **InteractE** run:

  ```shell
  python interacte.py --data FB15k-237 --gpu 0 --name fb15k_237_run
  ```
  - `data` indicates the dataset used for training the model. Other options are `WN18RR` and `YAGO3-10`.
  - `gpu` is the GPU used for training the model.
  - `name` is the provided name of the run which can be later used for restoring the model.
  - Execute `python interacte.py --help` for listing all the available options.

### Evaluating Pre-trained model:

* Execute `sh preprocess.sh` for extracting the datasets and setting up the environment. 

* Download the pre-trained model from [here](https://drive.google.com/open?id=1ffwqdnJFm1A14n7Cph7XVXX6TKY4BOv1) and place in `torch_saved` directory. 

* To restore and evaluate run:

  ```shell
  python interacte.py --data FB15k-237 --gpu 0 
  					--name fb15k_237_pretrained 
  					--restore --epoch 0
  ```

### Citation:
Please cite the following paper if you use this code in your work.

```bibtex
@article{interacte2020,
       author = {{Vashishth}, Shikhar and {Sanyal}, Soumya and {Nitin}, Vikram and
         {Agrawal}, Nilesh and {Talukdar}, Partha},
        title = "{InteractE: Improving Convolution-based Knowledge Graph Embeddings by Increasing Feature Interactions}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
         year = "2019",
        month = "Nov",
          eid = {arXiv:1911.00219},
        pages = {arXiv:1911.00219},
archivePrefix = {arXiv},
       eprint = {1911.00219},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv191100219V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

For any clarification, comments, or suggestions please create an issue or contact [shikhar@iisc.ac.in](http://shikhar-vashishth.github.io).