output_filename="${1:-run_jobs.sh}"
datasets="mslr yahoo"
folds="1 2 3 4 5"
ls="0.1 0.01 0.001 0.0001 0.00001 0.000001"
Ns="50 100 200 500 1000"
policies="logging target"
subsets="validation test"
Ks="5 10 20 30"
Ms="1000 10000 100000 1000000 10000000 10000001"

batch_size=8

> $output_filename

for ds in $datasets; do
  for N in $Ns; do
    for policy in $policies; do
      for fold in $folds; do
        for subset in $subsets; do
          echo "python mpl/experiments/propensities.py --params_path configs/propensities/propensities_RQ3_${ds}_${policy}.json --fold ${fold} --subset ${subset} --N ${N} --l 0.00001 --K 10 --method MPL --bs ${batch_size}" >> ${output_filename}
        done
      done
    done
  done
done

echo "Total jobs generated: $(wc -l < ${output_filename})"
chmod 755 ${output_filename}
