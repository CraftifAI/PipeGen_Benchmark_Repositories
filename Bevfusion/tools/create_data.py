import argparse
import os
from os import path as osp
from data_converter import nuscenes_converter as nuscenes_converter
from data_converter.create_gt_database import create_groundtruth_database

def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """Prepare data related to nuScenes dataset."""
    
    # Absolute paths bana lete hain taaki confusion na ho
    root_path = osp.abspath(root_path)
    out_dir = osp.abspath(out_dir)
    
    print(f"--- Step 1: Generating Info Files for {version} ---")
    print(f"Data Root: {root_path}")
    
    # 1. Info Files Generate karein
    # Note: Ab hum 'out_dir' pass nahi kar rahe kyunki converter khud root_path me save karega
    if load_augmented is None:
        nuscenes_converter.create_nuscenes_infos(
            root_path, 
            info_prefix, 
            version=version, 
            max_sweeps=max_sweeps
        )
    
    # 2. File Check logic
    # Kyunki ab converter sahi naam (nuscenes_infos_train.pkl) se save karega root_path mein
    train_pkl = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
    
    if not os.path.exists(train_pkl):
        # Fallback check: Agar galti se out_dir mein chali gayi ho
        alt_pkl = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")
        if os.path.exists(alt_pkl):
            train_pkl = alt_pkl
        else:
            print(f"CRITICAL ERROR: Info file not found at: {train_pkl}")
            print("Please check if Step 1 completed successfully.")
            return

    print(f"Found Info File: {train_pkl}")
    print(f"--- Step 2: Creating GT Database ---")

    # 3. GT Database Create karein
    create_groundtruth_database(
        dataset_name,
        root_path,
        info_prefix,
        train_pkl,
        load_augmented=load_augmented,
    )
    print("--- SUCCESS: Data preparation complete! ---")

parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument("--root-path", type=str, default="./data/kitti", help="specify the root path of dataset")
parser.add_argument("--version", type=str, default="v1.0", required=False, help="specify the dataset version")
parser.add_argument("--max-sweeps", type=int, default=10, required=False, help="specify sweeps of lidar per example")
parser.add_argument("--out-dir", type=str, default="./data/kitti", required=False, help="name of info pkl")
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument("--workers", type=int, default=4, help="number of threads to be used")
args = parser.parse_args()

if __name__ == "__main__":
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes":
        # Ab chahe Mini ho ya TrainVal, logic same rahega
        # Kyunki converter file ab smart hai
        
        if args.version == "v1.0-mini":
             nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,
                dataset_name="NuScenesDataset",
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                load_augmented=load_augmented,
            )
        else:
            train_version = f"{args.version}-trainval"
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name="NuScenesDataset",
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                load_augmented=load_augmented,
            )
            test_version = f"{args.version}-test"
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=test_version,
                dataset_name="NuScenesDataset",
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                load_augmented=load_augmented,
            )