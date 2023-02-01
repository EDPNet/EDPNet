# EDPNet
#you need to download xception_pytorch_imagenet.pth and put it into the folder,imitmodel

### Usage
1. Requirement:
The proposed EDPNet was trained and predicted based on deep learning server using an NVIDIA GTX 3090 24 GB GPU and a 3.00 GHz Intel Core i9-10980 XE CPU64GB main memory. We implement the programming based on Pytorch-gpu 1.7.1 deep learning framework in Python 3.6 language. The NVIDIA GTX 3090 GPU is powered by CUDA 11.1.
2. Train:
  - Download related datasets and symlink the paths to them as follows (you can alternatively modify the relevant paths specified in folder `config`):  
     ```
     cd semseg
     mkdir -p dataset
     ln -s /path_to_cityscapes_dataset dataset/cityscapes
     ```
   - Specify the gpu used in config then do training:

     ```shell
     sh tool/train.sh cityscapes EDPNet
     ```
3. Test:
 - Download trained segmentation models and put them under folder specified in config or modify the specified paths.

   - For full testing (get listed performance):

     ```shell
     sh tool/test.sh cityscapes EDPNet
     ```
