#!/bin/bash
N_ARGS=1

if [ "$#" -ne "${N_ARGS}" ]; then
  echo "Missing ${N_ARGS} args."
  exit 1
fi

start=${1}
end=${start} #${2}

# Go to each class directory
# Find the cluster with the largest number of files
dir="mean_images"

path_labels="synset_words.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

for unit in `seq ${start} ${end}`; do
#for unit in {0..999}; do
#for d in `ls -d ${dir}/n*`; 
  category=`echo ${labels[unit]} | cut -d " " -f 1`
  d="${dir}/${category}"

  for seed in {0..0}; do
    # fc8 params
    layer=fc8
    xy=0
    name="${layer}_${seed}"
    # Optimize images maximizing fc8 unit
    python ./act_max.tvd.center.py \
      --unit=${unit} \
      --filename=${name} \
      --layer=${layer} \
      --xy=${xy} \
      --seed=${seed}
  done
done
