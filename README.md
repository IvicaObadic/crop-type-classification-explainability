# Explaining Attention with Domain Knowledge
This repository provides the code for explanations of the attention weights with domain knowledge on the problem of crop-type classification. 
The crop-type classification dataset and the architecture of the Transformer Encoder model are based on the following repo: https://github.com/MarcCoru/crop-type-mapping. 

The training of the transformer encoder model with custom set of model hyperparameters and a choice of classes to be occluded can be performed by executing the following script: 
```
python train_and_evaluate_crop_type_classifier.py 
  --dataset_folder BavarianCrops/
  --num_classes 12
  --classes_to_exclude corn
  --results_root_dir /results
  --seq_aggr right_padding
  --pos_enc_opt seq_order
  --num_layers 1
  --num_heads 1
  --model_dim 128
  --save_weights_and_gradients
  ```
If the ```save_weights_and_gradients``` parameter is set to true, the attention weights of the field parcels in the test set are stored for further explainability analysis.

The code for the explanations of the learned attention weights patterns can be found in the jupyter notebooks inside the notebooks directory.
