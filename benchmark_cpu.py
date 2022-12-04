# 用于测试mmdetection模型的CPU推理速度

import os
import torch
import time
from argparse import ArgumentParser
from mmcv import Config
from mmcv.ops import RoIPool
from mmcv.parallel import collate
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.apis import init_detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./configs/yolact/yolact_r50_1x8_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='./work_dirs/yolact_r50_1x8_coco_4/epoch_62.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.config)
    imgs_path = cfg.data.test.img_prefix
    model = init_detector(args.config, args.checkpoint, device=args.device)

    num_warmup = 5
    pure_inf_time = 0
    for i, img_name in enumerate(os.listdir(imgs_path)):
        img_path = imgs_path + img_name
        model.cfg.data.test.pipeline = replace_ImageToTensor(model.cfg.data.test.pipeline)
        test_pipeline = Compose(model.cfg.data.test.pipeline)

        data = dict(img_info=dict(filename=img_path), img_prefix=None)
        data = test_pipeline(data)
        data = [data]
        data = collate(data, samples_per_gpu=1)
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

        start_time = time.perf_counter()
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Done image [{i + 1:<3}/ 100], fps: {fps:.1f} img / s')

        if (i + 1) == 100:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break



if __name__ == '__main__':
    args = parse_args()
    main(args)
