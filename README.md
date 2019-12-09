# GDP
Tensorflow implementation for GDP.

Accelerating convolutional neural networks has recently received ever-increasing research focus. Among various approaches proposed in the literature, ﬁlter pruning has been regarded as a promising solution, which is due to its advantage in signiﬁcant speedup and memory reduction of both network model and intermediate feature maps. To this end, most approaches tend to prune ﬁlters in a layerwise ﬁxed manner, which is incapable to dynamically recover the previously removed ﬁlter, as well as jointly optimize the pruned network across layers. In this paper, we propose a novel global & dynamic pruning (GDP) scheme to prune redundant ﬁlters for CNN acceleration. In particular, GDP ﬁrst globally prunes the unsalient ﬁlters across all layers by proposing a global discriminative function based on prior knowledge of each ﬁlter. Second, it dynamically updates the ﬁlter saliency all over the pruned sparse network, and then recovers the mistakenly pruned ﬁlter, followed by a retraining phase to improve the model accuracy. Specially, we effectively solve the corresponding nonconvex optimization problem of the proposed GDP via stochastic gradient descent with greedy alternative updating. Extensive experiments show that the proposed approach achieves superior performance to accelerate several cutting-edge CNNs on the ILSVRC 2012 benchmark, comparing to the state-of-the-art ﬁlter pruning methods.

## Citation
If you find our project useful in your research, please consider citing:

```
@article{lin2018accelerating,
  title={Accelerating Convolutional Networks via Global \& Dynamic Filter Pruning.},
  author={Lin, Shaohui and Ji, Rongrong and Li, Yuchao and Wu, Yongjian and Huang, Feiyue and Zhang, Baochang},
  journal={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2018}
}
```

## Running

The Dataset and Pre-Trained Model can be found in https://github.com/tensorflow/models/tree/master/research/slim.

### Training GDP (VGG-16 $\beta=0.7$)

```
python train.py --model_name vgg_16_gdp \
                --labels_offset 1 \
                --checkpoint_path {dir of vgg16 pretrained model} \
                --dataset_dir {dir of imagenet tfrecord} \
                --preprocessing_name vgg_16 \
                --train_image_size 224 \
                --train_dir tmp/vgg_16_gdp \
                --batch_size 32 \
                --learning_rate 0.001 \
                --end_learning_rate 0.00001 \
                --num_epochs_per_decay 10 \
                --max_number_of_steps 1200000 \
                --beta 0.7
```

### Finetune GDP 

```
python finetune.py --model_name vgg_16_gdp \
                --labels_offset 1 \
                --checkpoint_path tmp/vgg_16_gdp/model.ckpt-1200000 \
                --dataset_dir {dir of imagenet tfrecord} \
                --preprocessing_name vgg_16 \
                --train_image_size 224 \
                --train_dir tmp/vgg_16_gdp_ft \
                --batch_size 32 \
                --learning_rate 0.0001 \
                --end_learning_rate 0.00001 \
                --num_epochs_per_decay 10 \
                --max_number_of_steps 800000 
```

### Evaluation GDP

```
python eval.py --model_name vgg_16_gdp \
                --labels_offset 1 \
                --checkpoint_path tmp/vgg_16_gdp_ft/model.ckpt-800000 \
                --dataset_dir {dir of imagenet tfrecord} \
                --preprocessing_name vgg_16 \
                --eval_image_size 224
```

## Tips

If you find any problems, please feel free to contact to the authors (shaohuilin007@gmail.com or xiamenlyc@gmail.com).
