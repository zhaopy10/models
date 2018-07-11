# for training, run this command
# Note: remember to change the path of fine_tune_checkpoint path and batch size
# num_clones is set the same as the number of GPU used
# batch_size = num_clones * 8 (not sure, just recommendation)
python train.py --logtostderr --train_dir=./train_mobilenet_v2_only_person --pipeline_config_path=faster_rcnn_mobilenet_v2_even_smallerNet.configi --num_clones=4

# for evaluation, run this command
# run eval while training to monitor the change of mAP performance
# the pipeline.config file is automatically copied when run training
python eval.py --logtostderr --checkpoint_dir=./train_mobilenet_v2_only_person --eval_dir=./train_mobilenet_v2_only_person --pipeline_config_path=./train_mobilenet_v2_only_person/pipeline.config

# to export model, run this command
# use the newest checkpoint 
# pipeline_nms.config is a copy of pipeline.config, but iou_socre and iou_threshold are changed according to your selection
python export_inference_graph.py --input_type image_tensor --pipeline_config_path train_mobilenet_v2_only_person/pipeline_nms.config --trained_checkpoint_prefix train_mobilenet_v2_only_person/model.ckpt-187237 --output_directory export_mobilenet_v2
