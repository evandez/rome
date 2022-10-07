#!/bin/zsh
MODEL_NAME="gpt2"
DEVICE=5

cd ..
for DATASET in counterfact winoventi
do
    for MEDIATED in med umed
    do
        for TARGET in subj_first subj_last attr
        do
            KEY=${DATASET}_${MEDIATION}_${TARGET}
            CUDA_VISIBLE_DEVICES=$DEVICE python3 -m experiments.causal_trace \
                --model_name $MODEL_NAME \
                --fact_file data/mediation/${KEY}.json \
                --output_dir results/{model_name}/${KEY}/causal_trace
        done
    done
done
cd -
