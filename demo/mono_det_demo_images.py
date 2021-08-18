from argparse import ArgumentParser
import glob
from mmdet3d.apis import (inference_mono_3d_detector,inference_mono_detector, init_model,
                          show_result_meshlab)

import functools
import time
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument('imageDir', help='image folder')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show', action='store_true', help='show online visuliaztion results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visuliaztion results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    #run on images
    print("image dir is ", args.imageDir)
    input_dir = '/home/robolution-tech/data/realsense_road'
    image_list = glob.glob(args.imageDir+'/*')
    print(image_list)

    #result, data = inference_mono_3d_detector(model, args.image, args.ann)
    mono_detector = inference_mono_detector(model, args.ann)
    for image in image_list:
        t0 = time.time()
        result, data = mono_detector.inference_mono_3d_detector(image)
        t1 = time.time()
        # print(result)
        # show the results
        show_result_meshlab(
            data,
            result,
            args.out_dir,
            args.score_thr,
            show=args.show,
            snapshot=args.snapshot,
            task='mono-det')

        print(t1-t0)
if __name__ == '__main__':
    main()
