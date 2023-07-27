# Self-supervised Few-shot Learning for Semantic Segmentation: An Annotation-free Approach
This is the implementation of the paper "Self-supervised Few-shot Learning for Semantic Segmentation: An Annotation-free Approach" by Sanaz Karimijafarbigloo, Reza Azad, and Dorit Merhof. Implemented on Python 3.7 and Pytorch 1.5.1.

<p align="middle">
    <img src="data/git_visualize/architecture.png">
</p>

For more information, check out our paper on [[arXiv]()].

## Requirements

- Python 3.7
- PyTorch 1.5.1
- cuda 10.1

Conda environment settings:
```bash
conda create -n fewshot python=3.7
conda activate fewshot

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
```
## Preparing Few-Shot Segmentation Dataset
Download the following dataset:

> #### 1. FSS-1000
> Download FSS-1000 images and annotations from our [[Google Drive](https://drive.google.com/file/d/1Fn-cUESMMF1pQy8Xff-vPQvXJdZoUlP3/view?usp=sharing)].

Create a directory 'Dataset' and use the folder path in train/test arguments. 


## Generate sudo mask for test supports
In order to accelerate the inference process, we separated the spectral method for generating a support mask. please go to 'create_mask' and run the 'creating_mask.ipynb' notebook to create masks for all samples. In our data loader, we use these generated masks during the inference time for the test set. Hence, our method will predict the new samples without requiring support annotation. 
Download the following dataset:


## Training

> ### 1. FSS-1000
> ```bash
> python train.py --backbone resnet50
>                 --benchmark fss 
>                 --lr 1e-3
>                 --bsz 20
>                 --logpath "your_experiment_name"
> ```
> * Training takes approx. 12 hours (trained with RTX A5000 GPU).

## Testing

> ### 1. FSS-1000
> python test.py --backbone resnet50
>                --benchmark fss 
>                --nshot 1
>                --load "path_to_trained_model/weight.pt"
> ```

#### Example qualitative results:

<p align="middle">
    <img src="data/git_visualize/vis_results.png">
</p>
   
## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@InProceedings{fewshot
}
````
