#-*- coding:utf-8 -*-
import os
import sys
import argparse 
import numpy as np
from utils.opts import get_args
from utils.dataset.mycsvdataset import split_data
from utils.dataset.create_labels_train_val import create_train_val_data
from utils.dataset.dataset_builder import build_dataset,build_dataset_val,build_dataset_test
from utils.trainval import train_val
from utils.onnx.cvt2onnx import cvt2onnx
from utils.metrics import multiclass_metrics
import string





characters_set= string.digits + string.ascii_uppercase# +string.ascii_lowercase

if __name__ == '__main__':

    args = get_args()
    #'''
    num_classes = 2
    time_step =8
    feature_dims = 2
    train_data,val_data,test_data,feature_dims = split_data(time_step)
    

    #'''
    #'''
    train_data_loader,val_data_loader,test_data_loader = build_dataset(train_data,val_data,test_data,
                                                        num_classes = num_classes,batch_size=args.batch_size,
                                                        num_workers=args.workers)
    
    # best_acc,best_loss,best_epoch,results = train_val(train_data_loader=train_data_loader, 
    #                                                     val_data_loader=val_data_loader, 
    #                                                     train_num=len(train_data),
    #                                                     val_num=len(val_data),
    #                                                     num_classes=num_classes,args=args)

  
    cvt2onnx(args,time_step,feature_dims,num_classes)
    multiclass_metrics.plt_roc(args,data_loader=test_data_loader,num_classes=num_classes)

    