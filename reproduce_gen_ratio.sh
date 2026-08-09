
cd gen_ratio_profiler

python3 run_and_plot.py \
    --architecture decoder \
    --shape-name gpt3_2p7b \
    --seq-lens 128,256,512,1024 \
    --token-scenarios 1,10,20%,50%,100%

python3 run_and_plot.py \
    --architecture encoder_decoder \
    --shape-name bart_large \
    --seq-lens 128,256,512,1024 \
    --token-scenarios 1,10,20%,50%,100%

python3 plot_from_csv.py --architecture decoder --shape-name gpt3_2p7b
python3 plot_from_csv.py --architecture encoder_decoder --shape-name bart_large
