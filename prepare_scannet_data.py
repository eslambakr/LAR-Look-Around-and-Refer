import argparse
import pprint
import time
import os.path as osp
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
from referit3d.utils import immediate_subdirectories, create_dir, pickle_data, str2bool

"""
-top-scan-dir /home/e/scannet_dataset/scannet/scans -top-save-dir /home/e/scannet_dataset/scannet/scan_4_nr3d --load-dense False --save-jpg True --imgsize 128 --apply-global-alignment False
"""


def parse_args():
    parser = argparse.ArgumentParser(description='ReferIt3D')

    parser.add_argument('-top-scan-dir', required=True, type=str,
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('-top-save-dir', required=True, type=str,
                        help='the path of the directory to be saved preprocessed scans as a .pkl')

    # Optional arguments.
    parser.add_argument('--n-processes', default=-1, type=int,
                        help='the number of processes, -1 means use the available max')
    parser.add_argument('--process-only-zero-view', default=True, type=str2bool,
                        help='00_view of scans are used')
    parser.add_argument('--verbose', default=True, type=str2bool, help='')
    parser.add_argument('--apply-global-alignment', default=True, type=str2bool,
                        help='rotate/translate entire scan globally to aligned it with other scans')
    # Eslam
    parser.add_argument('--load-dense', default=False, type=str2bool, help='Load dense version of point-clouds')
    parser.add_argument('--save-jpg', default=False, type=str2bool, help='Save projected images directly')
    parser.add_argument('--imgsize', default=32, type=int, help='Size of projected image')
    parser.add_argument('--cocoon', default=False, type=str2bool, help='Save cocoon images for each object')
    parser.add_argument('--twoStreams', default=False, type=str2bool, help='Save 2d images for each object and raw pc')
    parser.add_argument('--geo', default=False, type=str2bool, help='Save 2d images for each object and raw pc')

    ret = parser.parse_args()

    # Print the args
    args_string = pprint.pformat(vars(ret))
    print(args_string)

    return ret


if __name__ == '__main__':
    args = parse_args()

    if args.process_only_zero_view:
        tag = 'keep_all_points_00_view'
    else:
        tag = 'keep_all_points'

    if args.apply_global_alignment:
        tag += '_with_global_scan_alignment'
    else:
        tag += '_no_global_scan_alignment'

    if args.load_dense:
        tag += '_densePCLoaded'
    if args.save_jpg:
        tag += '_saveJPG'
    if args.cocoon:
        tag += '_cocoon'
    if args.twoStreams:
        tag += '_twoStreams'
    if args.geo:
        tag += '_GEO'

    # Read all scan files.
    all_scan_ids = [osp.basename(i) for i in immediate_subdirectories(args.top_scan_dir)]
    print('{} scans found.'.format(len(all_scan_ids)))

    kept_scan_ids = []
    if args.process_only_zero_view:
        for si in all_scan_ids:
            if si.endswith('00'):
                kept_scan_ids.append(si)
        all_scan_ids = kept_scan_ids
    print('Working with {} scans.'.format(len(all_scan_ids)))
    if args.load_dense and False:
        list_all_scans = np.array_split(np.array(all_scan_ids), 7)
    else:
        list_all_scans = [all_scan_ids]

    # Prepare ScannetDataset
    idx_to_semantic_class_file = 'referit3d/data/mappings/scannet_idx_to_semantic_class.json'
    instance_class_to_semantic_class_file = 'referit3d/data/mappings/scannet_instance_class_to_semantic_class.json'
    axis_alignment_info_file = 'referit3d/data/scannet/scans_axis_alignment_matrices.json'

    scannet = ScannetDataset(args.top_scan_dir,
                             idx_to_semantic_class_file,
                             instance_class_to_semantic_class_file,
                             axis_alignment_info_file)

    def scannet_loader(scan_id):
        """Helper function to load the scans in memory.
        :param scan_id:
        :return: the loaded scan.
        """
        global scannet, args

        print("scan_id = ", scan_id)
        scan_i = ScannetScan(scan_id, scannet, args.apply_global_alignment, load_dense=args.load_dense,
                             save_jpg=args.save_jpg, img_size=args.imgsize, top_scan_dir=args.top_scan_dir,
                             cocoon=args.cocoon)
        if args.load_dense:
            scan_i.load_point_clouds_of_all_objects_dense()
        else:
            scan_i.load_point_clouds_of_all_objects()

        if args.save_jpg:
            scan_i.pc = None
            scan_i.color = None
        return scan_i

    if args.verbose:
        print('Loading scans in memory...')

    start_time = time.time()
    for id, all_scan_ids in enumerate(list_all_scans):
        print("Start processing of id = ", id)
        n_items = len(all_scan_ids)
        if args.n_processes == -1:
            n_processes = min(mp.cpu_count(), n_items)

        pool = mp.Pool(n_processes)
        chunks = int(n_items / n_processes)

        all_scans = dict()
        for i, data in enumerate(pool.imap(scannet_loader, all_scan_ids, chunksize=chunks)):
            all_scans[all_scan_ids[i]] = data

        pool.close()
        pool.join()

        if args.verbose:
            print("Loading raw data took {:.4} minutes.".format((time.time() - start_time) / 60.0))

        # Save data
        if args.verbose:
            print('Saving the results.')
        all_scans = list(all_scans.values())
        save_dir = create_dir(osp.join(args.top_save_dir, tag))
        if args.load_dense:
            save_file = osp.join(save_dir, tag + "_" + str(id) + '.pkl')
        else:
            save_file = osp.join(save_dir, tag + '.pkl')
        pickle_data(save_file, scannet, all_scans)
        print("Finish processing of id = ", id)

    if args.verbose:
        print('All done.')
