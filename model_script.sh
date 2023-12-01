# finetune
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=1234 train_edis.py \
--config ./configs/retrieval_edis.yaml \
--output_dir output/retrieval_edis_mblip_4gpus_5e-5

# evaluate
python evaluate_retrieval.py --config configs/retrieval_evaluate.yaml --image_bank restricted --cuda 0
python compute_metrics.py -d output/evaluate_results