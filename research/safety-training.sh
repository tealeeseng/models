export CUDA_VISIBLE_DEVICES="0"


python3 object_detection/model_main.py \
    --pipeline_config_path=safety-pipeline.config \
    --model_dir=safety-loads/models \
    --num_train_steps=1000 \
    --alsologtostderr

#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
