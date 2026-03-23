#!/usr/bin/env bash
set -e

# to temporarily test JAX091 release
pip install jax==0.9.1 jaxlib==0.9.1
pip install jax-rocm7-plugin
pip install jax-rocm7-pjrt

export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="

uv tree --depth 1 \
  | sed 's/[├└──]//g' \
  | awk '{print $1}' \
  | grep -viE '^(flax|jax|torch)\b' \
  | grep -viF '(*)' \
  > base-requirements.txt

grep -vi '^tensorflow' base-requirements.txt > /tmp/base.txt
{
  echo "tensorflow==2.19.1"
  echo "tensorflow-datasets"
  echo "importlib_resources"
} >> /tmp/base.txt
mv /tmp/base.txt base-requirements.txt

pip3 install -r base-requirements.txt

if [ "$1" = "convolution" ]; then
  cd examples/mnist
  python3 main.py --workdir=/tmp/mnist --config=configs/default.py 2>&1 | tee jax_convolution.log
  loss=$(grep "test_loss" jax_convolution.log | tail -1 | cut -d " " -f 8)
  echo "performance: $loss loss"

elif [ "$1" = "resnet" ]; then
  cp -f imagenette.py examples/imagenet/configs/imagenette.py && cd examples/imagenet
  sed -i '/dataset_builder = tfds.builder(config.dataset)/a\  dataset_builder.download_and_prepare()' train.py
  sed -i 's/if jax.process_index() == 0 and config.profile:/if jax.process_index() == 0 and hasattr(config, "profile") and config.profile:/' train.py
  python3 main.py --workdir="$PWD/imagenette" --config=configs/imagenette.py 2>&1 | tee log.txt
  loss=$(grep "eval epoch" log.txt | tail -1 | cut -d " " -f 9)
  echo "performance: $loss loss"

elif [ "$1" = "nlp_seq" ]; then
  cd examples/nlp_seq
  wget -O ud-treebanks-v2.0.tgz \
    https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1976/ud-treebanks-v2.0.tgz
  tar xzf ud-treebanks-v2.0.tgz

  python3 train.py \
    --batch_size=64 \
    --model_dir=./model_dir \
    --dev=ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-dev.conllu \
    --train=ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-train.conllu \
    2>&1 | tee jax_nlp_seq.log

  loss=$(grep "eval in step:" jax_nlp_seq.log | tail -1 | cut -d " " -f 10)
  echo "performance: $loss loss"

else
  echo "Invalid model name"
  exit 1
fi
