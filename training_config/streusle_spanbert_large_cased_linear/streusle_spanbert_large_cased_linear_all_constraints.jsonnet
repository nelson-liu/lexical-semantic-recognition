local transformer_model = "SpanBERT/spanbert-large-cased";
local max_length = 256;
local train_with_constraints = true;
local use_upos_constraints = true;
local use_lemma_constraints = true;
local use_predicted_upos = false;
local use_predicted_lemmas = false;
{
  "dataset_reader": {
    "type": "streusle",
    "use_predicted_upos": use_predicted_upos,
    "use_predicted_lemmas": use_predicted_lemmas,
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length
      },
    },
  },
  "train_data_path": "data/streusle/streusle.ud_train.json",
  "validation_data_path": "data/streusle/streusle.ud_dev.json",
  "test_data_path": "data/streusle/streusle.ud_test.json",
  "model": {
    "type": "streusle_tagger_linear",
    "train_with_constraints": train_with_constraints,
    "use_upos_constraints": use_upos_constraints,
    "use_lemma_constraints": use_lemma_constraints,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length
        }
      }
    },
  },
  "data_loader": {
    "batch_size": 64
  },
  "trainer": {
    "validation_metric": "+accuracy",
    "optimizer": {
        "type": "adam",
        "lr": 0.001,
        "parameter_groups": [
            [["^text_field_embedder.*$"], {"requires_grad": false}]
        ]
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
