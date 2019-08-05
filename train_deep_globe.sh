export CUDA_VISIBLE_DEVICES=0
python3 train_deep_globe.py \
--n_class 7 \
--data_path "./data/DeepGlobe_mod/" \
--model_path "./experiments/deepglobe/" \
--log_path "./experiments/deepglobe/" \
--task_name "fpn_global.508_4.28.2019_lr2e5" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_global.resize512_9.2.2018.2.global.pth" \
--path_g2l "fpn_global2local.508.deep.cat.1x_ensemble_fmreg.p3_10.14.2018.lr2e5.pth" \
--path_l2g "fpn_local2global.508_deep.cat_ensemble.p3_10.31.2018.lr2e5.local1x.epoch13.pth" \
--num_workers 8
# --path_g "cityscapes_global.800_4.5.2019.lr5e5.pth" \
# --path_g2l "fpn_global2local.508_deep.cat.1x_fmreg_ensemble.p3.0.15l2_3.19.2019.lr2e5.pth" \
# --path_l2g "fpn_local2global.508_deep.cat.1x_fmreg_ensemble.p3_3.19.2019.lr2e5.pth" \
