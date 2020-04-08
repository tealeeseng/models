export CUDA_VISIBLE_DEVICES="0"


python3 object_detection/model_main.py \
    --pipeline_config_path=object_detection/safety-large-loads/safety-larger-pipeline.config \
    --model_dir=object_detection/safety-large-loads/models \
    --num_train_steps=10000 \
    --alsologtostderr

#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
