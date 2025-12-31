# rename_mp4.py
import os, argparse, glob

def run(folder, prefix, start_idx, digits, dry):
    files = sorted(glob.glob(os.path.join(folder, '*.mp4')))
    if not files:
        print('未找到 MP4 文件')
        return
    fmt = f'{{:0{digits}d}}'
    for i, old in enumerate(files, start=start_idx):
        base_name = prefix + fmt.format(i) + '.mp4'
        new = os.path.join(folder, base_name)
        print(f'{os.path.basename(old)}  ->  {base_name}')
        if not dry:
            os.rename(old, new)
    print('完成！' if not dry else '（预览模式，未真正改名）')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量重命名 MP4')
    parser.add_argument('--dir', default='.', help='目标文件夹')
    parser.add_argument('--prefix', default='', help='新文件名前缀')
    parser.add_argument('--start', type=int, default=1, help='起始序号')
    parser.add_argument('--digits', type=int, default=3, help='序号位数')
    parser.add_argument('--dry', action='store_true', help='只预览不执行')
    args = parser.parse_args()
    run(args.dir, args.prefix, args.start, args.digits, args.dry)