import torch,monai
import sys,os,json
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
from datautils import resampleVolume,adjust_image_direction
from MPUM import MPUM
from tqdm import tqdm
from collections import OrderedDict
current_dir = os.path.dirname(os.path.abspath(__file__))
def resample_image(input_image, reference_image):
    def dataset_config_to_rulematrix(x):
        with open(x,"r") as f:
            x = json.load(f)
        rules = torch.zeros(len(x),215)
        for key in x:
            rules[x[key]["labelindex"]][x[key]["modelindex"]] = 1
        rules[0] = 1 - torch.sum(rules[1:],dim=0)
        return rules
    
    # 创建一个重采样过滤器
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    
    # 设置插值方法，通常使用线性插值
    resampler.SetInterpolator(sitk.sitkLinear)
    
    # 设置输出使用参考图像的spacing, origin和direction
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    
    # 设置输出图像的size
    resampler.SetSize(reference_image.GetSize())
    resampler.SetDefaultPixelValue(0)
    
    # 执行重采样
    return resampler.Execute(input_image)

# def inference(config, nii_path,output_seg_path,output_stdct_path=None,check=True,modelname=None,dataset_mapping_to_model=None):
#     '''
#         dataset_mapping_to_model: .json file path
#     '''
#     ct_path = nii_path
#     assert "." not in output_seg_path, "output_seg_path should be a dir path, not a file path"
    
#     print(f"提供的NIfTI路径是:{ct_path}")
#     # file_path = os.path.dirname(os.path.abspath(__file__))
    
#     orict_itk = sitk.ReadImage(os.path.join(ct_path))
#     ct_itk = sitk.ReadImage(os.path.join(ct_path))
    
#     print("----------------direction check and spacing check------------------------")

#     print("before processing, spacing:",ct_itk.GetSpacing())
#     print("before processing, direction:",ct_itk.GetDirection())
    
#     new_direction = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
#     new_spacing = (2,2,2)  # 保持间距不变
    
#     direction_check = np.mean(np.abs(np.array(ct_itk.GetDirection()) - np.array(new_direction)))
#     spacing_check = np.mean(np.abs(np.array(ct_itk.GetSpacing()) - np.array([2,2,2])))

#     print("----------------pre-process <LUCID Standard Protocol>------------------------")
    
#     ct_itk = resampleVolume(new_spacing,ct_itk,resamplemethod=sitk.sitkLinear)
#     ct_itk = adjust_image_direction(ct_itk, new_direction)

#     print("after processing, spacing:",ct_itk.GetSpacing())
#     print("after processing, direction:",ct_itk.GetDirection())

#     if output_stdct_path is not None:
#         output_stdct_path_ = os.path.dirname(output_stdct_path)
#         if not os.path.exists(output_stdct_path_):
#             os.makedirs(output_stdct_path_)
#             print(f"目录已创建：{output_stdct_path_}")
#         sitk.WriteImage(ct_itk, output_stdct_path)
#         print("standard protocol nii has been write in ",output_stdct_path)
#     else:
#         print("if need to save CT.nii.gz file in standard protocol (1.5mm), use arg <output_stdct_path>")
    
#     def scale_intensity_range(ct, a_min, a_max, b_min, b_max, clip):
#         if clip:
#             ct = torch.clamp(ct, min=a_min, max=a_max)
#         # 线性缩放
#         ct = (ct - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
#         return ct
        
#     ct = sitk.GetArrayFromImage(ct_itk)
#     ct = torch.tensor(ct).float().unsqueeze(0).unsqueeze(0)
#     print(ct.shape)
#     if config["modality"] == 'ct':
#         ct = scale_intensity_range(ct, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
#     elif config["modality"] == 'pet':
#         ct = scale_intensity_range(ct, a_min=0, a_max=20, b_min=0.0, b_max=1.0, clip=True)
#     elif config["modality"] == "mr":
#         ct = scale_intensity_range(ct, a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True)
    
#     print("single model mode!!")
#     tissueclip = torch.load("../tissueclip_RN101.pth").to("cuda:0")
#     print(config)
#     model = MPUM(config,tissueclip,1)

#     print(config["ckpt"])
#     ckpt = torch.load(config["ckpt"],map_location="cpu")
    
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in ckpt['model'].items():  # 假设权重存储在'ckpt['model']'中
#         name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)
    
#     model = model.to("cuda:0")
#     model = model.eval()
#     print(ct.shape)
#     print("----------------sliding_window_inference------------------------")
#     with torch.no_grad():
#         wb_pred = monai.inferers.sliding_window_inference(
#                     ct,(128,128,128),
#                     sw_batch_size=1,
#                     predictor=model,
#                     overlap=0.5,
#                     mode="gaussian",
#                     sw_device="cuda:0",
#                     device="cpu",
#                     progress=True)
#         # wb_pred = torch.sigmoid(wb_pred)
#     print("----------------post-process------------------------")
    
#     if dataset_mapping_to_model is not None:
#         dataset_to_model_config = dataset_config_to_rulematrix(dataset_mapping_to_model)
#         b,c,z,w,y = wb_pred.shape
#         pred = torch.matmul(dataset_to_model_config.float(), wb_pred.view(b,c,-1)).view(b,-1,z,w,y)
#         for ii,value in enumerate(torch.sum(dataset_to_model_config,dim=1)):
#             if value == 1:
#                 continue
#             pred[0,ii],_ = torch.max(wb_pred[0][dataset_to_model_config[ii]==1],dim=0)
#         wb_pred = pred.float()
#         with open(dataset_mapping_to_model,"r") as f:
#             d = json.load(f)
#         readme = {}
#         for key in d:
#             readme[len(readme)] = key
#     else:
#         from categories import prediction as label215
#         readme = label215

#     combined = torch.argmax(wb_pred[0],dim=0).detach().cpu()
#     sitk_image = sitk.GetImageFromArray(combined)
#     # 设置方向和像素间距
#     sitk_image.SetDirection(ct_itk.GetDirection())
#     sitk_image.SetSpacing(ct_itk.GetSpacing())
#     sitk_image.SetOrigin(ct_itk.GetOrigin())

#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(orict_itk)
#     resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#     sitk_image = resampler.Execute(sitk_image)
#     array = sitk.GetArrayFromImage(sitk_image)
#     print(np.max(array))
#     print(array.shape)
#     if not os.path.exists(output_seg_path):
#         os.makedirs(output_seg_path)
#         print(f"目录已创建：{output_seg_path}")
#     sitk.WriteImage(sitk_image, os.path.join(output_seg_path,"merge.nii.gz"))
#     print("create nii.gz ",os.path.join(output_seg_path,"merge.nii.gz"))
#     with open(os.path.join(output_seg_path,"readme.json"),"w") as f:
#         json.dump(readme,f)

def inference(config, nii_path,output_seg_path,output_stdct_path=None,check=True,modelname=None,dataset_mapping_to_model=None):
    '''
        dataset_mapping_to_model: .json file path
    '''
    config["tissuenumber"] = 215
    ct_path = nii_path
    # assert "." not in output_seg_path, "output_seg_path should be a dir path, not a file path"
    
    print(f"提供的NIfTI路径是:{ct_path}")
    # file_path = os.path.dirname(os.path.abspath(__file__))
    
    orict_itk = sitk.ReadImage(os.path.join(ct_path))
    ct_itk = sitk.ReadImage(os.path.join(ct_path))
    
    print("----------------direction check and spacing check------------------------")

    print("before processing, spacing:",ct_itk.GetSpacing())
    print("before processing, direction:",ct_itk.GetDirection())
    
    new_direction = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
    new_spacing = (2,2,2)  # 保持间距不变
    
    direction_check = np.mean(np.abs(np.array(ct_itk.GetDirection()) - np.array(new_direction)))
    spacing_check = np.mean(np.abs(np.array(ct_itk.GetSpacing()) - np.array([2,2,2])))

    print("----------------pre-process <LUCID Standard Protocol>------------------------")
    
    ct_itk = resampleVolume(new_spacing,ct_itk,resamplemethod=sitk.sitkLinear)
    ct_itk = adjust_image_direction(ct_itk, new_direction)

    print("after processing, spacing:",ct_itk.GetSpacing())
    print("after processing, direction:",ct_itk.GetDirection())

    if output_stdct_path is not None:
        output_stdct_path_ = os.path.dirname(output_stdct_path)
        if not os.path.exists(output_stdct_path_):
            os.makedirs(output_stdct_path_)
            print(f"目录已创建：{output_stdct_path_}")
        sitk.WriteImage(ct_itk, output_stdct_path)
        print("standard protocol nii has been write in ",output_stdct_path)
    else:
        print("if need to save CT.nii.gz file in standard protocol (1.5mm), use arg <output_stdct_path>")
    
    def scale_intensity_range(ct, a_min, a_max, b_min, b_max, clip):
        if clip:
            ct = torch.clamp(ct, min=a_min, max=a_max)
        # 线性缩放
        ct = (ct - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
        return ct
        
    ct = sitk.GetArrayFromImage(ct_itk)
    ct = torch.tensor(ct).float().unsqueeze(0).unsqueeze(0)
    print(ct.shape)
    if config["modality"] == 'ct':
        ct = scale_intensity_range(ct, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
    elif config["modality"] == 'pet':
        ct = scale_intensity_range(ct, a_min=0, a_max=20, b_min=0.0, b_max=1.0, clip=True)
    elif config["modality"] == "mr":
        ct = scale_intensity_range(ct, a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True)

    if isinstance(config["ckpt"],list):
        print("multi model mode!!")
        wb_preds = []
        for modelnum,oneckpt in enumerate(config["ckpt"]):
            print("model {} inference...".format(modelnum))
            tissueclip = torch.load(os.path.join(current_dir,"tissueclip_RN101.pth")).to("cuda:0")
            model = MPUM(config,tissueclip,1)
            ckpt = torch.load(oneckpt,map_location="cpu")
            new_state_dict = OrderedDict()
            for k, v in ckpt['model'].items():  # 假设权重存储在'ckpt['model']'中
                name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model = model.to("cuda:0")
            model = model.eval()
            print(ct.shape)
            print("----------------sliding_window_inference------------------------")
            with torch.no_grad():
                wb_pred = monai.inferers.sliding_window_inference(
                            ct,(128,128,128),
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.5,
                            mode="gaussian",
                            sw_device="cuda:0",
                            device="cpu",
                            progress=True)
                wb_pred = torch.sigmoid(wb_pred)
                if config["tissue"] == "all":
                    pass
                elif config["tissue"] == "brain":
                    wb_pred[:,1:132] = 0
                wb_preds.append(wb_pred)
        wb_pred = 0
        for www in wb_preds:
            wb_pred += www

        if dataset_mapping_to_model is not None:
            dataset_to_model_config = dataset_config_to_rulematrix(dataset_mapping_to_model)
            b,c,z,w,y = wb_pred.shape
            pred = torch.matmul(dataset_to_model_config.float(), wb_pred.view(b,c,-1)).view(b,-1,z,w,y)
            for ii,value in enumerate(torch.sum(dataset_to_model_config,dim=1)):
                if value == 1:
                    continue
                pred[0,ii],_ = torch.max(wb_pred[0][dataset_to_model_config[ii]==1],dim=0)
            wb_pred = pred.float()
            with open(dataset_mapping_to_model,"r") as f:
                d = json.load(f)
            readme = {}
            for key in d:
                readme[len(readme)] = key
        else:
            from categories import prediction as label215
            readme = label215

        combined = torch.argmax(wb_pred[0],dim=0).detach().cpu()
        sitk_image = sitk.GetImageFromArray(combined)
        # 设置方向和像素间距
        sitk_image.SetDirection(ct_itk.GetDirection())
        sitk_image.SetSpacing(ct_itk.GetSpacing())
        sitk_image.SetOrigin(ct_itk.GetOrigin())
    
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(orict_itk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        sitk_image = resampler.Execute(sitk_image)
        array = sitk.GetArrayFromImage(sitk_image)
        print(np.max(array))
        print(array.shape)
        if not os.path.exists(output_seg_path):
            os.makedirs(output_seg_path)
            print(f"目录已创建：{output_seg_path}")
        sitk.WriteImage(sitk_image, os.path.join(output_seg_path,"merge.nii.gz"))
        print("create nii.gz ",os.path.join(output_seg_path,"merge.nii.gz"))
    else:
        print("single model mode!!")
        tissueclip = torch.load(os.path.join(current_dir,"tissueclip_RN101.pth")).to("cuda:0")
        print(config)
        model = MPUM(config,tissueclip,1)
    
        print(config["ckpt"])
        ckpt = torch.load(config["ckpt"],map_location="cpu")
        
        new_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():  # 假设权重存储在'ckpt['model']'中
            name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        model = model.to("cuda:0")
        model = model.eval()
        print(ct.shape)
        print("----------------sliding_window_inference------------------------")
        with torch.no_grad():
            wb_pred = monai.inferers.sliding_window_inference(
                        ct,(128,128,128),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        mode="gaussian",
                        sw_device="cuda:0",
                        device="cpu",
                        progress=True)
            # wb_pred = torch.sigmoid(wb_pred)
        print("----------------post-process------------------------")
        
        if dataset_mapping_to_model is not None:
            dataset_to_model_config = dataset_config_to_rulematrix(dataset_mapping_to_model)
            b,c,z,w,y = wb_pred.shape
            pred = torch.matmul(dataset_to_model_config.float(), wb_pred.view(b,c,-1)).view(b,-1,z,w,y)
            for ii,value in enumerate(torch.sum(dataset_to_model_config,dim=1)):
                if value == 1:
                    continue
                pred[0,ii],_ = torch.max(wb_pred[0][dataset_to_model_config[ii]==1],dim=0)
            wb_pred = pred.float()
            with open(dataset_mapping_to_model,"r") as f:
                d = json.load(f)
            readme = {}
            for key in d:
                readme[len(readme)] = key
        else:
            from categories import prediction as label215
            readme = label215
    
        combined = torch.argmax(wb_pred[0],dim=0).detach().cpu()
        sitk_image = sitk.GetImageFromArray(combined)
        # 设置方向和像素间距
        sitk_image.SetDirection(ct_itk.GetDirection())
        sitk_image.SetSpacing(ct_itk.GetSpacing())
        sitk_image.SetOrigin(ct_itk.GetOrigin())
    
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(orict_itk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        sitk_image = resampler.Execute(sitk_image)
        array = sitk.GetArrayFromImage(sitk_image)
        print(np.max(array))
        print(array.shape)
        if not os.path.exists(output_seg_path):
            os.makedirs(output_seg_path)
            print(f"目录已创建：{output_seg_path}")
        sitk.WriteImage(sitk_image, os.path.join(output_seg_path,"merge.nii.gz"))
        print("create nii.gz ",os.path.join(output_seg_path,"merge.nii.gz"))
    with open(os.path.join(output_seg_path,"readme.json"),"w") as f:
        json.dump(readme,f)