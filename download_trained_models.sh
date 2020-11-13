#!/usr/bin/env bash

download_folder(){
  local model_link=$1
  local model_name=$2
  wget -c "${model_link}/${model_name}.zip"
  unzip "${model_name}".zip
  rm "${model_name}".zip
  mv "${model_name}"/* .
  rm -rf "${model_name}"
}

download_folder "http://download941.mediafire.com/iwao6d1x1big/vqqiabataunkczs" "softmax-models"
download_folder "http://download941.mediafire.com/am3y3dtntvag/eb994fafkjx2fvk" "entmax15-models"
download_folder "http://download941.mediafire.com/ntz61qnupaeg/jjczy5dvp7nt6ih" "sparsemax-models"
