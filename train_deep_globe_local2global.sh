export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--n_class 7 \
--data_path "/ssd1/chenwy/deep_globe/data/" \
--model_path "/home/chenwy/deep_globe/saved_models/" \
--log_path "/home/chenwy/deep_globe/runs/" \
--task_name "fpn_deepglobe_local2global" \
--mode 3 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_deepglobe_global.pth" \
--path_g2l "fpn_deepglobe_global2local.pth" \