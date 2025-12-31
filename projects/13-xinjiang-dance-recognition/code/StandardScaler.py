"""
preprocess_norm.py
训练集单独 fit → 同一变换复用（MinMax 或 StandardScaler）
python StandardScaler.py --dir npy --out norm --mode minmax --split 0.2
"""
import os, json, glob, joblib, numpy as np, argparse, tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def fit_scaler(file_list, mode):
    """流式 fit，避免一次性 load 全部内存爆炸"""
    scaler = MinMaxScaler() if mode == 'minmax' else StandardScaler()
    for f in tqdm.tqdm(file_list, desc='Fit scaler'):
        x = np.load(f)                      # (F, 99)
        scaler.partial_fit(x)               # 增量更新
    return scaler

def transform_set(file_list, scaler, out_dir):
    """逐文件变换并保存"""
    os.makedirs(out_dir, exist_ok=True)
    log = []
    for f in tqdm.tqdm(file_list, desc=f'Transform {os.path.basename(out_dir)}'):
        x = np.load(f)                                   # (F, 99)
        x_norm = scaler.transform(x)*2                # 同一变换
        new_name = os.path.basename(f)                   # 保留原名
        out_path = os.path.join(out_dir, new_name)
        np.save(out_path, x_norm.astype(np.float32))
        log.append({'old': f, 'new': out_path})
    return log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='npy', help='原始 .npy 目录')
    parser.add_argument('--out', default='norm', help='输出根目录')
    parser.add_argument('--mode', choices=['minmax', 'std'], default='minmax', help='归一化方式')
    parser.add_argument('--split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(args.dir, '*.npy')))
    if not all_files:
        print('[WARN] 目录里没有 .npy 文件')
        return

    # 1. 拆分训练/验证（文件名层面 stratify）
    labels = [int(os.path.basename(f)[0]) for f in all_files]   # 文件名第1位=标签
    train_files, val_files = train_test_split(
        all_files, test_size=args.split, random_state=args.seed,
        stratify=labels)

    # 2. 只在训练集上拟合
    scaler = fit_scaler(train_files, args.mode)
    joblib.dump(scaler, os.path.join(args.out, f'{args.mode}_scaler.pkl'))

    # 3. 同一变换复用：训练集 + 验证集
    train_log = transform_set(train_files, scaler, os.path.join(args.out, 'train'))
    val_log   = transform_set(val_files,   scaler, os.path.join(args.out, 'val'))

    # 4. 保存映射日志（可逆）
    meta = {'mode': args.mode, 'split': args.split, 'train_cnt': len(train_files),
            'val_cnt': len(val_files), 'train_log': train_log, 'val_log': val_log}
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'✔ 完成！输出目录: {args.out}')
    print(f'  训练集: {len(train_files)} 个   验证集: {len(val_files)} 个')
    print(f'  scaler 已保存: {args.out}/{args.mode}_scaler.pkl')


if __name__ == '__main__':
    main()