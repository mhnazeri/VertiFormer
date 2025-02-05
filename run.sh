#! /usr/bin/env bash

export PROJECT_NAME=vertiformer  # add your project folder to python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_NAME
export COMET_LOGGING_CONSOLE=info
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # adds 24M overhead to memory usage
export CUDA_VISIBLE_DEVICES=0
# for torch.compile debugging
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1

Help()
{
   # Display Help
   echo 
   echo "Facilitates running different stages of training and evaluation."
   echo 
   echo "options:"
   echo "train_former                      Starts training VertiFormer."
   echo "train_encoder                     Starts training VertiEncoder."
   echo "train_decoder                     Starts training VertiDecoder."
   echo "train_dt                          Starts training Downstream task on top of VertiEncoder."
   echo "train_vanilla                     Starts training End-to-End Downstream."
   echo
}

run () {
  case $1 in
    train_encoder)
      python $PROJECT_NAME/train_vertiencoder.py --conf $PROJECT_NAME/conf/vertiencoder
      ;;
    train_former)
      python $PROJECT_NAME/train_vertiformer.py --conf $PROJECT_NAME/conf/vertiformer
      ;;
    train_decoder)
      python $PROJECT_NAME/train_vertidecoder.py --conf $PROJECT_NAME/conf/vertidecoder
      ;;
    train_dt)
      python $PROJECT_NAME/train_dt.py --conf $PROJECT_NAME/conf/dt
      ;;
    -h) # display Help
      Help
      exit
      ;;
    *)
      echo "Unknown '$1' argument. Please run with '-h' argument to see more details."
      # Help
      exit
      ;;
  esac
}

run $1 $2
