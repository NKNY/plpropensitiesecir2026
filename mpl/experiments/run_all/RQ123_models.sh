output_filename="${1:-run_jobs.sh}"
datasets="mslr yahoo"
folds="1 2 3 4 5"
ls="0.1 0.01 0.001 0.0001 0.00001 0.000001"
Ns="50 100 200 500 1000"
policies="logging target"
subsets="validation test"
Ks="5 10 20 30"
Ms="1000 10000 100000 1000000 10000000 10000001"

> $output_filename

for ds in $datasets; do
  for fold in $folds; do
    if [[ "$ds" == "mslr" ]]; then
      echo "python mpl/training/train.py --params_path configs/models/train_RQ12_mslr.json --fold ${fold}" >> ${output_filename}
    fi
    echo "python mpl/training/train.py --params_path configs/models/train_RQ3_${ds}_logging.json --fold ${fold}" >> ${output_filename}
    echo "python mpl/training/train.py --params_path configs/models/train_RQ3_${ds}_target.json --fold ${fold}" >> ${output_filename}
  done
done

echo "Total jobs generated: $(wc -l < ${output_filename})"
chmod 755 ${output_filename}
