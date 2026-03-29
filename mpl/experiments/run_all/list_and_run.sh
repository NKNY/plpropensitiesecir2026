#!/bin/bash
all_runs_filename="${1:-run_jobs.sh}"
job_gen_filename="${2:-RQ1_MPL.sh}"
$job_gen_filename $all_runs_filename

num_jobs=$(wc -l < "$all_runs_filename")

# Run all runs
$all_runs_filename