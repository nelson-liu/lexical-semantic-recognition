Training GloVe models:

```bash
allennlp train training_config/streusle_glove_bilstm_linear/streusle_glove_bilstm_linear_all_constraints_predicted_no_train_constraints.jsonnet --include-package streusle_tagger -s models/streusle_glove_bilstm_linear_all_constraints_predicted_no_train_constraints
allennlp train training_config/streusle_glove_bilstm_linear/streusle_glove_bilstm_linear_all_constraints_no_train_constraints.jsonnet --include-package streusle_tagger -s models/streusle_glove_bilstm_linear_all_constraints_no_train_constraints
```

Training BERT models:

```bash
for model in streusle_bert_large_cased_bilstm_linear streusle_bert_large_cased_linear; do 
    for i in all_constraints all_constraints_no_train_constraints all_constraints_predicted all_constraints_predicted_no_train_constraints no_constraints upos_constraints upos_constraints_no_train_constraints upos_constraints_predicted upos_constraints_predicted_no_train_constraints;
    do 
        allennlp train training_config/${model}/${model}_${i}.jsonnet \
            --include-package streusle_tagger \
            -s models/${model}_${i} ; 
    done;
done
```

Transferring to Google Drive:

```bash
for i in streusle_bert*; do 
    rclone copy ${i} "stanford_gdrive:Stanford/1/STREUSLE Tagging/models/${i}" y--progress;
done
```
