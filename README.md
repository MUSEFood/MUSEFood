# MUSEFood

## Code
We use Faster-RCNN and Grabcut to generate segmentation labels to pre-train our FCN. You can use the same method (i.e. use generated labels to pretrain and use manually annotated labels to fine-tune) to train a more accurate food image segmentation model.

The MutitaskFCN folder includes three multi-task FCN models proposed in our paper (i.e. MFCN-A to MFCN-C).

## SUEC Food Dataset
The SUEC Food dataset contains 600 segmented food images annotated manually and 31395 segmented food images annotated by Grabcut. All these images are from UEC Food-256 dataset. The SUEC Food dataset can be downloaded at https://drive.google.com/open?id=162GUja37w17aPKi19CXpmHcB38pwZXaf.

Note that this dataset can be used only for non-commercial research purpose. If you like to use it for any other purpose, please contact us.

## Citation
If you publish a paper using our food dataset, we'd glad if you could refer to the following paper:

@article{gao2019musefood,
  title={MUSEFood: Multi-sensor-based Food Volume Estimation on Smartphones},
  author={Gao, Junyi and Tan, Weihao and Ma, Liantao and Wang, Yasha and Tang, Wen},
  journal={arXiv preprint arXiv:1903.07437},
  year={2019}
}

And the original UEC Food-256 paper:

@InProceedings{kawano14c,
 author="Kawano, Y. and Yanai, K.",
 title="Automatic Expansion of a Food Image Dataset Leveraging Existing Categories with Domain Adaptation",
 booktitle="Proc. of ECCV Workshop on Transferring and Adapting Source
Knowledge in Computer Vision (TASK-CV)",
 year="2014",
}
