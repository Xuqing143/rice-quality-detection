import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from skimage import measure
from cellpose import models, io, utils, transforms, core
import os


# 你的数据集路径
img_dir = './samples/original/'
model_path = './models/cpsam'

flow_threshold = 0.4
cellprob_threshold = 0.0

# 匹配所有常见格式的图像
files = sorted(glob.glob(img_dir + "*.png") +
               glob.glob(img_dir + "*.jpg") +
               glob.glob(img_dir + "*.jpeg") +
               glob.glob(img_dir + "*.tif"))

print(f"找到 {len(files)} 张图片")
if len(files) == 0:
    raise FileNotFoundError("没有找到图像,请检查路径和文件后缀!")

# 读取图像
imgs = [io.imread(f) for f in files]


# 检查 GPU
if not core.use_gpu():
    raise ImportError("没有检测到GPU,建议在GPU环境运行")

# 加载 Cellpose-SAM 模型(你可以换成 "sam" 或 "cyto3" 等预训练权重)
model = models.CellposeModel(gpu=True,pretrained_model=model_path)

masks_pred, flows, styles = model.eval(
    imgs,
    niter=250,
    flow_threshold=flow_threshold,
    cellprob_threshold=cellprob_threshold
)

# 指定图像保存目录
save_dir = "./samples/results"
os.makedirs(save_dir, exist_ok=True)

#指定分割结果目录
results_dir = "./samples/csv"
os.makedirs(results_dir, exist_ok=True)

#存储所有图像的特征数据
all_props_data=[]

for i, (img,mask) in enumerate(zip(imgs,masks_pred)):
    # 每张图单独画（防止叠加混乱）
    fig, ax = plt.subplots(figsize=(4, 4))

    # 归一化原图
    if img.ndim == 3:
        show_img = img
    else:
        show_img = transforms.normalize_img(img, axis=0)

    ax.imshow(show_img, cmap="gray")

    # 画预测 mask 边界(黄色)
    outlines_pred = utils.outlines_list(masks_pred[i])
    for o in outlines_pred:
        ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=0.75, ls="--")

    ax.set_title(f"Image {i + 1}")
    ax.axis("off")

    # 保存为 PNG 文件
    out_path = os.path.join(save_dir, f"overlay_{i + 1:04d}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300, pil_kwargs={"compression": "none"})
    plt.close(fig)  # 关闭图像，防止内存占用
    print(out_path + "已保存")

    #生成每张图片的特征表格
    if mask.max() > 0:#确保检测到大米
        props = measure.regionprops_table(mask, properties=['label',
                                                            'centroid',
                                                            'area',
                                                            'axis_major_length',
                                                            'axis_minor_length'])
        #创建DateFrame并计算长宽比
        props_df=pd.DataFrame(props)
        props_df['aspect_ratio'] = props_df['axis_major_length'] / props_df['axis_minor_length']

        #添加图像编号
        props_df['image_id']=i+1
        props_df['image_name']=os.path.basename(files[i])



        #添加大米数量到最后一行
        rice_count=len(props_df)
        summary_row={
            'label':'总计',
            'centroid-0':'',
            'centroid-1':'',
            'area':props_df['area'].sum(),
            'axis_major_length':props_df['axis_major_length'].mean(),
            'axis_minor_length':props_df['axis_minor_length'].mean(),
            'aspect_ratio': props_df['aspect_ratio'].mean(),
            'image_id': i + 1,
            'image_name':os.path.basename(files[i]),
            'rice_count':rice_count
        }

        #创建包含所有数据和汇总行的完整表格
        final_df=props_df.copy()
        final_df['rice_count']=''#普通行不显示数量
        summary_df=pd.DataFrame([summary_row])
        final_df=pd.concat([final_df,summary_df],ignore_index=True)

        #保存单张图片的特征表格
        singel_csv_path=os.path.join(results_dir, f"rice_properties_image_{i+1:04d}.csv")
        final_df.to_csv(singel_csv_path, index=False,encoding='utf-8-sig')
        print(f"图像{i+1}的特征表格已保存：{singel_csv_path}")
        print(f"图像{i+1}的大米数量：{rice_count}")

        #为总表收集数据
        props_df['rice_count']=rice_count
        all_props_data.append(props_df)
    else:
        print(f"图像{i+1}未识别到大米")
        empty_df=pd.DataFrame({
            'label': '总计',
            'centroid-0': '',
            'centroid-1': '',
            'area': [0],
            'axis_major_length': [0],
            'axis_minor_length': [0],
            'aspect_ratio': [0],
            'image_id': [i + 1],
            'image_name': os.path.basename(files[i]),
            'rice_count': [0]
        })
        empty_csv_path=os.path.join(results_dir, f"rice_properties_image_{i+1:04d}.csv")
        empty_df.to_csv(empty_csv_path, index=False,encoding='utf-8-sig')
        print(f"图像{i + 1}的特征表格已保存：{empty_csv_path}")

print(f"所有带边界的结果图已保存到：{save_dir}")

#生成所有图像的综合特征表格
if all_props_data:
    all_props_df=pd.concat(all_props_data, ignore_index=True)

    #保存综合表格
    combined_csv_path=os.path.join(results_dir, f"all_rice_properties.csv")
    all_props_df.to_csv(combined_csv_path, index=False,encoding='utf-8-sig')
    print(f"所有大米的综合特征表格已保存：{combined_csv_path}")

    #计算大米数量
    total_rice=sum([df['rice_count'].iloc[0] for df in all_props_data])
    print(f"总大米数量是：{total_rice}")
else:
    print("未在任何图像中检测到大米")