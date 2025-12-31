"""
rename_serial.py
把文件夹里所有 .npy 依次重命名为 0001.npy、0002.npy...
python rename_serial.py --dir npy --digits 4 --dry
"""
import argparse, glob, os, tqdm, json, pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='npy', help='目标目录')
    parser.add_argument('--digits', type=int, default=4, help='序号位数')
    parser.add_argument('--dry', action='store_true', help='只预览不执行')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, '*.npy')))
    if not files:
        print('[WARN] 目录里没有 .npy 文件')
        return

    fmt = f'{{:0{args.digits}d}}.npy'          # 例如 0001.npy
    log = []                                    # 备份日志
    for idx, old_path in enumerate(tqdm.tqdm(files, desc='Rename'), 1):
        new_name = fmt.format(idx)
        new_path = os.path.join(args.dir, new_name)
        log.append({'old': os.path.basename(old_path), 'new': new_name})
        if args.dry:
            print(f'[DRY]  {os.path.basename(old_path)}  ->  {new_name}')
        else:
            os.rename(old_path, new_path)

    # 保存重命名日志（可逆）
    log_file = os.path.join(args.dir, 'rename_log.json')
    if not args.dry:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    print(f'✔ 完成{"（预览）" if args.dry else ""}，日志写入 {log_file}')

if __name__ == '__main__':
    main()