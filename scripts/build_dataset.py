import os
import glob
import shutil


titmouse_path = '/media/mana/mana/dataset/no_watertight_sampling'
titmouse_items = sorted(glob.glob(os.path.join(titmouse_path, '*')))
print(titmouse_items)

titmouse_pifu_path = '/media/mana/mana/dataset/Titmouse_pifu/SAMPLES_WO_WT'

for item in titmouse_items[1:100]:
    pcd_path = os.path.join(item, 'sample')
    # pcd_item = glob.glob(pcd_path)[-1]
    nums = item[-4:]
    print(nums)

    target_path = os.path.join(titmouse_pifu_path, str(nums))
    # if not os.path.exists(target_path):
        # os.mkdir(target_path)
    shutil.copytree(pcd_path, target_path)
