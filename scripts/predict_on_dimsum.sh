for modelpath in models/streusle*; do
    modelname=$(basename ${modelpath})
    echo "Evaluating ${modelname}"
    allennlp predict models/${modelname}/model.tar.gz data/dimsum16/dimsum16.test.blind.jsonl \
        --silent \
        --output-file models/${modelname}/${modelname}_dimsum16_test_predictions.jsonl \
        --silent \
        --include-package streusle_tagger \
        --predictor streusle-tagger \
        --batch-size 64
done
