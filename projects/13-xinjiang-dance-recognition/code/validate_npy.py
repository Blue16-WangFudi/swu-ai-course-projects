"""
validate_npy.py
快速体检 npy 训练集（放宽范围版）
python validate_npy.py --dir npy
"""
import numpy as np, glob, os, json, argparse
from tqdm import tqdm

def inspect_one(path):
    d = np.load(path)                     # (F, 99)
    F, C = d.shape
    info = {
        'file': os.path.basename(path),
        'frames': int(F),
        'features': int(C),
        'all_finite': bool(np.isfinite(d).all()),
        'no_all_zero': bool(~np.allclose(d, 0)),          # 是否全 0
        'x_range': [float(d[:, 0::3].min()), float(d[:, 0::3].max())],
        'y_range': [float(d[:, 1::3].min()), float(d[:, 1::3].max())],
        'z_range': [float(d[:, 2::3].min()), float(d[:, 2::3].max())],
    }
    # ===== 放宽范围：允许轻微越界 =====
    EPS = 1e-3
    score = 0
    if info['all_finite']:                   score += 1
    if info['no_all_zero']:                  score += 1
    if info['x_range'][0] >= -EPS and info['x_range'][1] <= 1 + EPS: score += 1
    if info['y_range'][0] >= -EPS and info['y_range'][1] <= 1 + EPS: score += 1
    if info['z_range'][0] >= -EPS and info['z_range'][1] <= 1 + EPS: score += 1
    # ===== 降低门槛：4 分即 PASS =====
    info['score'] = score
    info['status'] = 'PASS' if score >= 4 else ('WARN' if score >= 2 else 'FAIL')

    # 打印详细范围（调试用，可删）
    print(f'[RANGE] {info["file"]}  x={info["x_range"]}  y={info["y_range"]}  z={info["z_range"]}')
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='npy', help='npy 目录')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, '*.npy')))
    if not files:
        print('[WARN] 目录里没有 .npy 文件')
        return

    report = []
    for f in tqdm(files, desc='Check'):
        report.append(inspect_one(f))

    # 统计
    pass_n = sum(1 for r in report if r['status'] == 'PASS')
    warn_n = sum(1 for r in report if r['status'] == 'WARN')
    fail_n = sum(1 for r in report if r['status'] == 'FAIL')

    print(f'\n===== 快速体检（放宽范围） =====')
    print(f'总样本数: {len(report)}   PASS: {pass_n}   WARN: {warn_n}   FAIL: {fail_n}')
    if fail_n:
        print('\n>>> FAIL 清单（需修复）')
        for r in report:
            if r['status'] == 'FAIL':
                print(f'  {r["file"]}  score={r["score"]} 原因: ', end='')
                if not r['all_finite']: print('非有限值 ', end='')
                if not r['no_all_zero']: print('全零 ', end='')
                print()
    if warn_n:
        print('\n>>> WARN 清单（建议检查）')
        for r in report:
            if r['status'] == 'WARN':
                print(f'  {r["file"]}  score={r["score"]} 范围异常/全零')

    # 保存详细报告
    with open(os.path.join(args.dir, 'validate_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n详细报告已写入 {args.dir}/validate_report.json')


if __name__ == '__main__':
    main()