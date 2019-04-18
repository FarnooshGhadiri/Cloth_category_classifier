#!/bin/bash

#if [ "$#" -ne 1 ]; then
#  echo "Usage: env.sh <cluster>"
#  exit 0
#fi

env

CLUSTER="mila"
if [ $CLUSTER == "mila" ]; then
  export TMP_DIR=/tmp/beckhamc/deep_fashion
  DF_IMG=/network/data1/beckhamc/deep_fashion/DF_Img_Low.zip
  DF_ANNO=/network/data1/beckhamc/deep_fashion/DF_Anno.zip
  export RESULTS_DIR=results_mixup
elif [ $CLUSTER == "cc" ]; then
  #TMP_DIR=${SLURM_TMPDIR}
  #CELEBA_DATASET=/home/cjb60/project/beckhamc/data/celeba/txt
  #CELEBA_ZIP=/home/cjb60/project/beckhamc/data/celeba/img_align_celeba.zip
  #export RESULTS_DIR=/scratch/cjb60/github/swapgan_mixup/results_mixup
  echo 'not implemented yet'
  exit 0
else
  echo 'Error: arg must be either mila or cc'
  exit 0
fi

echo 'TMP_DIR = ' ${TMP_DIR}
echo 'DF_IMG = ' ${DF_IMG}
echo 'DF_ANNO = ' ${DF_ANNO}

if [ ! -d ${TMP_DIR} ]; then
  echo ${TMP_DIR} ' does not exist so mkdir...'
  mkdir -p ${TMP_DIR}
fi

echo 'copying images...'
if [ ! -d ${TMP_DIR}/DF_Img_Low ]; then
  scp ${DF_IMG} ${TMP_DIR}
  `cd ${TMP_DIR} && unzip -o DF_Img_Low.zip`
fi

echo 'copying annotations...'
if [ ! -d ${TMP_DIR}/Anno ]; then
  scp ${DF_ANNO} ${TMP_DIR}
  `cd ${TMP_DIR} && unzip -o DF_Anno.zip`
fi
