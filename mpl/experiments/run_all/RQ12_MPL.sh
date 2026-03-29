output_filename="${1:-run_jobs.sh}"
datasets="mslr yahoo"
folds="1 2 3 4 5"
ls="0.1 0.01 0.001 0.0001 0.00001 0.000001"
Ns="50 100 200 500 1000"
policies="logging target"
subsets="validation test"
Ks="5 10 20 30"
Ms="1000 10000 100000 1000000 10000000 10000001"

declare -A batch_sizes

batch_sizes["K10,N1000"]=8
batch_sizes["K20,N1000"]=8
batch_sizes["K20,N1000"]=4
batch_sizes["K30,N200"]=8
batch_sizes["K30,N500"]=4
batch_sizes["K30,N1000"]=2

> $output_filename

# RQ1 US
seed=0
for K in $Ks; do
  for N in $Ns; do
    for fold in $folds; do
      for l in $ls; do
        if [[ $K != 10 && $l != 0.00001 ]]; then
          continue
        fi
        batch_size_key="K$K,N$N"
        if [[ -v batch_sizes[$batch_size_key] ]]; then
          batch_size_str="--bs ${batch_sizes[$batch_size_key]}"
        fi
        echo "python mpl/experiments/propensities.py --params_path configs/propensities/propensities_RQ12_mslr.json --fold ${fold} --subset test --N ${N} --l ${l} --K ${K} --method MPL ${batch_size_str}" >> ${output_filename}
      done
    done
  done
done

echo "Total jobs generated: $(wc -l < ${output_filename})"
chmod 755 ${output_filename}
