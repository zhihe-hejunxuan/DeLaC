import os
import argparse
from runner import Runner
from paths import resolve_split_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='result/mp_20/', help='The directory for storing training outputs')
    parser.add_argument('--dataset', type=str, default='mp_20', help='Dataset name, must be perov_5, carbon_24, or mp_20')
    parser.add_argument(
        '--data_root',
        type=str,
        default='',
        help='mp_20 时默认指向父项目 ../data/mp_20；其它数据集默认 SyMat/data/<dataset>',
    )
    parser.add_argument(
        '--score_norm',
        type=str,
        default='',
        help='覆盖 score_norm 路径；mp_20 默认使用父项目 data/mp_20/new_score_norm.txt',
    )
    args = parser.parse_args()

    result_path = args.result_path
    if not os.path.isdir(result_path):
        os.makedirs(result_path, exist_ok=True)

    assert args.dataset in ['perov_5', 'carbon_24', 'mp_20'], "Not supported dataset"

    paths = resolve_split_paths(args.dataset, data_root=args.data_root or None)
    train_data_path = paths['train_path']
    val_data_path = paths['val_path']
    score_norm_path = args.score_norm.strip() or paths['score_norm_path']

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] train={train_data_path}")
    print(f"[INFO] val={val_data_path}")
    print(f"[INFO] score_norm={score_norm_path}")
    print(f"[INFO] local_pt_dir={paths['pt_dir']}")

    if args.dataset == 'perov_5':
        from config.perov_5_config_dict import conf
    elif args.dataset == 'carbon_24':
        from config.carbon_24_config_dict import conf
    else:
        from config.mp_20_config_dict import conf

    runner = Runner(conf, score_norm_path)
    runner.train(
        train_data_path,
        val_data_path,
        result_path,
        train_pt_path=paths['train_pt_path'],
        val_pt_path=paths['val_pt_path'],
    )


if __name__ == '__main__':
    main()
