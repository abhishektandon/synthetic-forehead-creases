# Synthetic Forehead-Creases Biometric Generation for Reliable User Verification 
#### [Abhishek Tandon<sup>1</sup>](https://scholar.google.com/citations?user=0sXfNaQAAAAJ&hl=en), [Geetanjali Sharma<sup>1</sup>](https://scholar.google.com/citations?hl=en&user=Np8VOOAAAAAJ&view_op=list_works&sortby=pubdate), [Gaurav Jaswal<sup>2</sup>](https://scholar.google.co.in/citations?user=otGsksUAAAAJ&hl=en), [Aditya Nigam<sup>1</sup>](https://faculty.iitmandi.ac.in/~aditya/), [Raghavendra Ramachandra<sup>3</sup>](https://scholar.google.com/citations?user=OIYIrmIAAAAJ&hl=en)

##### <sup>1</sup>Indian Institute of Technology (IIT), Mandi, India, <sup>2</sup>Technology and Innovation Hub (TIH), IIT Mandi, India, <sup>3</sup>Norwegian University of Science and Technology (NTNU), Norway
--------
Generative AI for Futuristic Biometrics - IJCB'24 Special Session

[Arxiv Pre-print](https://arxiv.org/abs/2408.15693)

![main-figure](./imgs/main-figure.png)


# Results

| Database | EER | TMR@FMR = 0.1/0.01 (%) | Dataset | Pretrained Model |
|----------|:----------:|:----------:|:----------:|:----------:|
| FH-V1 (Real)  | 12.39     | 40.19/21.97 |  [link](https://ktiwari.in/biometrics/databases/)    |    [link](https://huggingface.co/abhi-td/synthetic-forehead-creases/tree/main/recognition/forehead-v1-adaface) |
| SS-PermuteAug (Synthetic)  | 9.38     | 60.32/45.68 |    [link](https://huggingface.co/datasets/abhi-td/synthetic-forehead-creases/tree/main/subject_specific_synthetic_datasets/ss-permute-aug)      |    [link](https://huggingface.co/abhi-td/synthetic-forehead-creases/tree/main/recognition/ss_permute_aug_adaface) |

----------
#### repository is under updation

# Citation
```
@article{tandon2024synthetic,
  title={Synthetic Forehead-creases Biometric Generation for Reliable User Verification},
  author={Tandon, Abhishek and Sharma, Geetanjali and Jaswal, Gaurav and Nigam, Aditya and Ramachandra, Raghavendra},
  journal={arXiv preprint arXiv:2408.15693},
  year={2024}
}
```