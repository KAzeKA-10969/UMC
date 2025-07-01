#!/usr/bin/bash
cd..

for seed in 0 1 2 3 4
do
    for multimodal_method in 'umc'
    do
        for method in 'umc'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in  'MIntRec' 
                do
                    python "C:/Machine_Learning/UMC/run.py" \
                    --dataset $dataset \
                    --data_path 'C:/Machine_Learning/UMC/Datasets/MIntRec' 
                    --logger_name $method \
                    --multimodal_method $multimodal_method \
                    --method $method\
                    --train "C:/Machine_Learning/UMC/Datasets/MIntRec/train.tsv"\
                    --tune \
                    --save_results \
                    --seed $seed \
                    --gpu_id '1' \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone  bert-base-uncased \
                    --config_file_name ${method}_${dataset} \
                    --results_file_name "results_umc.csv" \
                    --output_path "outputs/${dataset}"
                done
            done
        done
    done
done
