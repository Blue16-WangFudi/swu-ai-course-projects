import os

empty_files = [
    "data/sequences/nonviolence/NV_341.npy",
    "data/sequences/nonviolence/NV_365.npy",
    "data/sequences/nonviolence/NV_387.npy",
    "data/sequences/nonviolence/NV_393.npy",
    "data/sequences/nonviolence/NV_402.npy",
    "data/sequences/nonviolence/NV_415.npy",
    "data/sequences/nonviolence/NV_444.npy",
]

for f in empty_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除空文件: {f}")
