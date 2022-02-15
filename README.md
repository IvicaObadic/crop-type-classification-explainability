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
  --save_key_queries_embeddings
  ```
After the training of the model is completed, this script also evaluates the model performance on the test set. Setting the ```save_weights_and_gradients``` ```save_key_queries_embeddings``` parameters saves the attention weights and the key and query embeddings of the field parcels in the test set of the trained model for further explainability analysis.

These attention weights and their corresponding key and query embeddings are afterwards used to explain the learned attention patterns by performing attention sensitivity analysis and clustering the attention keys in order to discover the attention signature. The jupyter notebooks for these explainability analysis can be found under the ```notebooks``` directory.
