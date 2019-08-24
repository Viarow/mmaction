import argparse
import numpy as np
import pickle
import cv2
import os
import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)


def single_test(model, data_loader):
    model.eval()
    results = []
    inputs = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        inputs.append(data['img_group_0'].data[0].squeeze())
        for _ in range(batch_size):
            prog_bar.update()
        
        #print("\n")
        #print("*************************")
        #print(len(data['img_group_0'].data))
        #print(data['img_group_0'].data[0].size())
        
    return results, inputs


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    args = parser.parse_args()
    return args
    
    
def get_top_5_index(results_path, video_num):
    results = pickle.load(open(results_path,'rb'),encoding='utf-8')
    result = results[video_num]
    sorted_score = np.sort(result)
    sorted_index = np.argsort(result)
    top_scores = sorted_score[0][-5:]
    top_index = sorted_index[0][-5:]

    top_index = np.flip(top_index)
    return top_index
    

def returnCAM(feat_conv, weight_softmax, class_idx):
    size_upsample = (224, 224)        # decide on the input_size in tsn_rgb_bninception.py
    bz, nc, h, w = feat_conv.shape
    output_cam = []
    idx = class_idx[0]
    for snippet in range(0, bz):
        features = feat_conv[snippet]
        cam = weight_softmax[idx].dot(features.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
    
    
def numpy_2_image(array):
    channels = []
    for i in range(0, 3):
        layer = array[i]
        layer = layer - np.min(layer)
        channel = layer / np.max(layer)
        channel = np.uint8(255 * channel)
        channels.append(channel)
    img = cv2.merge(channels)
    return img
    
    
def writeCAMs(class_name, CAMs, imgs, video_idx):
    bz, nc, h, w = imgs.shape
    assert len(CAMs) == bz
    for snippet in range(0, bz):
        img = imgs[snippet]
        img = numpy_2_image(img)
        CAM = CAMs[snippet]
        heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('data/CAM_imgs/'+ class_name +'/CAMs_{:02d}/cam_{:03d}.jpg'.format(video_idx, snippet), result)
       
        
        
def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_recognizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint, strict=True)
        model = MMDataParallel(model, device_ids=[1])
        
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())       # fully conneted layer parameters to numpy already

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs, inputs = single_test(model, data_loader)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(recognizers, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)
            
    #print(len(features_blobs))
    #print(features_blobs[0].size())

    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)

    num_videos = len(outputs)
    class_name = 'YoYo'
    os.mkdir('data/CAM_imgs/' + class_name)

    for k in range(0, num_videos):
        os.mkdir('data/CAM_imgs/'+ class_name + '/CAMs_{:02d}'.format(k))
        idx = get_top_5_index("tools/results.pkl", k)  # change the dir of results.pkl to tools/
        conv_feat = pickle.load(open("tools/hook_features/feat_{:02d}.pkl".format(k), 'rb'), encoding='utf-8')
        conv_feat = conv_feat.cpu().numpy()
        CAMs = returnCAM(conv_feat, weight_softmax,
                         [idx[0]])  # generate class activation mapping for the top1 prediction
        single_input = inputs[k].numpy()
        writeCAMs(class_name, CAMs, single_input, k)

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann['label'])

    if args.use_softmax:
        print("Averaging score over {} clips with softmax".format(
            outputs[0].shape[0]))
        results = [softmax(res, dim=1).mean(axis=0) for res in outputs]
    else:
        print("Averaging score over {} clips without softmax (ie, raw)".format(
            outputs[0].shape[0]))
        results = [res.mean(axis=0) for res in outputs]
    top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
    mean_acc = mean_class_accuracy(results, gt_labels)
    print("Mean Class Accuracy = {:.02f}".format(mean_acc * 100))
    print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
    print("Top-5 Accuracy = {:.02f}".format(top5 * 100))
    # print("*********model._modules.keys**********")
    # print(model._modules.keys())
    # print("*********model.children**********")
    # print(list(model.children()))
    # print("*********model's parameters**********")
    # for name, param in model.named_parameters():
    #     print(name, param.data.size())


if __name__ == '__main__':
    main()
