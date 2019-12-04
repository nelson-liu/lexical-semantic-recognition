{
  "dataset_reader": {
    "type": "streusle",
    "use_predicted_upos": true,
    "token_indexers": {
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-large-cased"
        }
    }
  },
  "train_data_path": "data/streusle/streusle.ud_train.json",
  "validation_data_path": "data/streusle/streusle.ud_dev.json",
  "test_data_path": "data/streusle/streusle.ud_test.json",
  "model": {
    "type": "streusle_tagger",
    "train_with_constraints": false,
    "use_upos_constraints": true,
    "use_lemma_constraints": false,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"]
        },
        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-large-cased",
                "top_layer_only": false,
                "requires_grad": false
            }
        }
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "validation_metric": "+accuracy",
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "num_serialized_models_to_keep": 1,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
