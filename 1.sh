# python main.py --imb_ratio 50  --bs --gpu 0 > hesso_base_50.log 2>&1 &
# python main.py --imb_ratio 100 --bs --gpu 0 > hesso_base_100.log 2>&1 &

# python main.py --hesso --imb_ratio 50  --bs --gpu 2 > hesso_50.log 2>&1 &
# python main.py --hesso --imb_ratio 100 --bs --gpu 1 > hesso_100.log 2>&1 &

# python main.py --hesso --isc LT-vote --imb_ratio 50  --bs --gpu 1 > hesso_ours_50.log 2>&1 &
# python main.py --hesso --isc LT-vote --imb_ratio 100 --bs --gpu 7 > hesso_ours_100.log 2>&1 &

# python main.py --cagrad --imb_ratio 50  --ce --gpu 0 > cagrad_50.log 2>&1 &
# python main.py --cagrad --imb_ratio 100  --ce --gpu 0 > cagrad_100.log 2>&1 &

# python main_smaple.py --cagrad --imb_ratio 50  --ce --gpu 0 > cagrad_smaple_50.log 2>&1 &
# python main_smaple.py --cagrad --imb_ratio 100  --ce --gpu 0 > cagrad_smaple_100.log 2>&1 &

# python main.py --hesso --isc LT-vote --cagrad --imb_ratio 50  --ce --gpu 2 > ours_cagrad_50.log 2>&1 &
# python main.py --hesso --isc LT-vote --cagrad --imb_ratio 100  --ce --gpu 2 > ours_cagrad_100.log 2>&1 &

# python main_smaple.py --cagrad --hesso --isc LT-vote --imb_ratio 50  --ce --gpu 0 > ours_cagrad_smaple_50.log 2>&1 &
# python main_smaple.py --cagrad --hesso --isc LT-vote --imb_ratio 100  --ce --gpu 0 > ours_cagrad_smaple_100.log 2>&1 &

# # ---------------------------------------------------------------------------------------------------------------------
# python main.py --save_all_epoch --hesso --target_group_sparsity 0.1 --isc LT-vote --imb_ratio 100  --bs --gpu 1 > bs_1.log  2>&1 &
# python main.py --save_all_epoch --hesso --target_group_sparsity 0.3 --isc LT-vote --imb_ratio 100  --bs --gpu 3 > bs_3.log 2>&1 &
# python main.py --save_all_epoch --hesso --target_group_sparsity 0.5 --isc LT-vote --imb_ratio 100  --bs --gpu 3 > bs_5.log 2>&1 &
# python main.py --save_all_epoch --hesso --target_group_sparsity 0.7 --isc LT-vote --imb_ratio 100  --bs --gpu 5 > bs_7.log  2>&1 &
# python main.py --save_all_epoch --hesso --target_group_sparsity 0.9 --isc LT-vote --imb_ratio 100  --bs --gpu 5 > bs_9.log 2>&1 &

# python main.py --save_all_epoch --hesso --target_group_sparsity 0.7 --isc LT-vote --imb_ratio 100  --ce --gpu 5 > ce_50.log  2>&1 &

python main_ATO.py --imb_ratio 100  --bs --gpu 0 > ATO_bs_100.log  2>&1 &
python main_ATO.py --imb_ratio 50  --bs --gpu 4 > ATO_bs_50.log  2>&1 &