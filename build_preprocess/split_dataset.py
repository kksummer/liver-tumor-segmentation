import os
import random
import shutil


def count_slices_per_patient(patient_dir):
    patient_slices_count = {}
    for filename in os.listdir(patient_dir):
        patient_id = filename.split('_')[0]
        patient_slices_count[patient_id] = patient_slices_count.get(patient_id, 0) + 1
    return patient_slices_count


def split_data_cls2(dataset_dir, image_name='patient', mask_name='liver'):
    # 定义目录路径
    patient_dir = os.path.join(dataset_dir, image_name)
    mask_dir = os.path.join(dataset_dir, mask_name)

    # 计算每个病例的切片数量
    patient_slices_count = count_slices_per_patient(patient_dir)

    # 根据切片数量排序病例
    sorted_patients = sorted(patient_slices_count, key=patient_slices_count.get, reverse=True)

    # 根据切片数量分割
    total_slices = sum(patient_slices_count.values())
    target_train_slices = total_slices * 0.8
    current_train_slices = 0
    train_patient_ids = []
    for patient_id in sorted_patients:
        if current_train_slices < target_train_slices:
            train_patient_ids.append(patient_id)
            current_train_slices += patient_slices_count[patient_id]
        else:
            break

    # 剩余病例分配到验证集
    val_patient_ids = [pid for pid in patient_slices_count if pid not in train_patient_ids]

    # 创建训练集和验证集的目录结构
    train_patient_dir = os.path.join(dataset_dir, 'train', image_name)
    train_mask_dir = os.path.join(dataset_dir, 'train', mask_name)
    val_patient_dir = os.path.join(dataset_dir, 'val', image_name)
    val_mask_dir = os.path.join(dataset_dir, 'val', mask_name)

    # 确保目录存在
    os.makedirs(train_patient_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_patient_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    # 将文件复制到新的目录结构
    for filename in os.listdir(patient_dir):
        patient_id = filename.split('_')[0]
        if patient_id in train_patient_ids:
            shutil.move(os.path.join(patient_dir, filename), os.path.join(train_patient_dir, filename))
        else:
            shutil.move(os.path.join(patient_dir, filename), os.path.join(val_patient_dir, filename))

    for filename in os.listdir(mask_dir):
        patient_id = filename.split('_')[0]
        if patient_id in train_patient_ids:
            shutil.move(os.path.join(mask_dir, filename), os.path.join(train_mask_dir, filename))
        else:
            shutil.move(os.path.join(mask_dir, filename), os.path.join(val_mask_dir, filename))

    # 删除原始文件夹，如果它们为空
    if not os.listdir(patient_dir):
        os.rmdir(patient_dir)
    if not os.listdir(mask_dir):
        os.rmdir(mask_dir)


def split_data_cls3(dataset_dir, image_name='patient_tumor', mask_name='liver_wt', mask_name2='tumor'):
    # 定义目录路径
    patient_dir = os.path.join(dataset_dir, image_name)
    liver_mask_dir = os.path.join(dataset_dir, mask_name)
    tumor_mask_dir = os.path.join(dataset_dir, mask_name2)

    # 计算每个病例的切片数量
    patient_slices_count = count_slices_per_patient(patient_dir)

    # 根据切片数量排序病例
    sorted_patients = sorted(patient_slices_count, key=patient_slices_count.get, reverse=True)

    # 根据切片数量分割
    total_slices = sum(patient_slices_count.values())
    target_train_slices = total_slices * 0.8
    current_train_slices = 0
    train_patient_ids = []
    for patient_id in sorted_patients:
        if current_train_slices < target_train_slices:
            train_patient_ids.append(patient_id)
            current_train_slices += patient_slices_count[patient_id]
        else:
            break

    # 剩余病例分配到验证集
    val_patient_ids = [pid for pid in patient_slices_count if pid not in train_patient_ids]

    # 创建训练集和验证集的目录结构
    train_patient_dir = os.path.join(dataset_dir, 'train', image_name)
    train_liver_mask_dir = os.path.join(dataset_dir, 'train', mask_name)
    train_tumor_mask_dir = os.path.join(dataset_dir, 'train', mask_name2)
    val_patient_dir = os.path.join(dataset_dir, 'val', image_name)
    val_liver_mask_dir = os.path.join(dataset_dir, 'val', mask_name)
    val_tumor_mask_dir = os.path.join(dataset_dir, 'val', mask_name2)

    # 确保目录存在
    os.makedirs(train_patient_dir, exist_ok=True)
    os.makedirs(train_liver_mask_dir, exist_ok=True)
    os.makedirs(train_tumor_mask_dir, exist_ok=True)
    os.makedirs(val_patient_dir, exist_ok=True)
    os.makedirs(val_liver_mask_dir, exist_ok=True)
    os.makedirs(val_tumor_mask_dir, exist_ok=True)

    # 将文件复制到新的目录结构
    for filename in os.listdir(patient_dir):
        patient_id = filename.split('_')[0]
        if patient_id in train_patient_ids:
            shutil.move(os.path.join(patient_dir, filename), os.path.join(train_patient_dir, filename))
        else:
            shutil.move(os.path.join(patient_dir, filename), os.path.join(val_patient_dir, filename))

    for filename in os.listdir(liver_mask_dir):
        patient_id = filename.split('_')[0]
        if patient_id in train_patient_ids:
            shutil.move(os.path.join(liver_mask_dir, filename), os.path.join(train_liver_mask_dir, filename))
        else:
            shutil.move(os.path.join(liver_mask_dir, filename), os.path.join(val_liver_mask_dir, filename))

    for filename in os.listdir(tumor_mask_dir):
        patient_id = filename.split('_')[0]
        if patient_id in train_patient_ids:
            shutil.move(os.path.join(tumor_mask_dir, filename), os.path.join(train_tumor_mask_dir, filename))
        else:
            shutil.move(os.path.join(tumor_mask_dir, filename), os.path.join(val_tumor_mask_dir, filename))

    # 删除原始文件夹，如果它们为空
    if not os.listdir(patient_dir):
        os.rmdir(patient_dir)
    if not os.listdir(liver_mask_dir):
        os.rmdir(liver_mask_dir)
    if not os.listdir(tumor_mask_dir):
        os.rmdir(tumor_mask_dir)