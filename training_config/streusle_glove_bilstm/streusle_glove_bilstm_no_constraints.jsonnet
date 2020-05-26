{
  "dataset_reader": {
    "type": "streusle",
    "use_predicted_upos": true,
    "use_predicted_lemmas": true,
    "token_indexers": {
        "tokens": {
            "type": "single_id"
        },
        "token_characters": {
            "type": "characters",
            "min_padding_length": 5
        }
    }
  },
  "train_data_path": "data/streusle/streusle.ud_train.json",
  "validation_data_path": "data/streusle/streusle.ud_dev.json",
  "test_data_path": "data/streusle/streusle.ud_test.json",
  "model": {
    "type": "streusle_tagger",
    "use_upos_constraints": false,
    "use_lemma_constraints": false,
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                "embedding_dim": 300,
                "trainable": false
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 64
                },
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 64,
                    "num_filters": 200,
                    "ngram_filter_sizes": [
                        5
                    ]
                }
            }
        }
    },
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 500,
      "hidden_size": 256,
      "num_layers": 2
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
