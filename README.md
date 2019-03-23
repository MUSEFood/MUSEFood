# MUSEFood
the Faster-RCNN+Grabcut uses Faster-RCNN and Grabcut to generate annotations fed to FCN.

the MutitaskFCN includes three FCN models with different deep architectures to segement food item.

The SUEC Food dataset contains 600 segmented food images annotated manually and 31395 segmented food images annotated by Grabcut. All these images are from UEC Food-258 dataset. The SUEC Food dataset can be downloaded at https://drive.google.com/open?id=162GUja37w17aPKi19CXpmHcB38pwZXaf.

Note that this dataset can be used only for non-commercial research purpose. If you like to use it for any other purpose, please contact us.

If you publish a paper using our food dataset, we'd glad if you could refer to the following paper:

@article{gao2019multi,
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
