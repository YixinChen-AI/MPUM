from itertools import product

import numpy as np
import SimpleITK as sitk


def adjust_image_direction(image, desired_direction, verbose=True, *, output_spacing=None, default_value=0):
    """Orient a 3-D image while keeping its voxel values in physical space."""
    desired_direction = tuple(desired_direction)
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(desired_direction)
    oriented = sitk.DICOMOrient(image, orientation)
    if np.allclose(oriented.GetDirection(), desired_direction, rtol=0, atol=1e-6):
        if verbose:
            print(f"Image axes oriented to {orientation}")
        if output_spacing is None:
            return oriented
        return resampleVolume(output_spacing, oriented, resamplemethod=sitk.sitkLinear)

    # Axis permutations cannot make an oblique image cardinal. Resample its
    # full voxel extent into the requested direction using physical coordinates.
    axes = np.asarray(desired_direction).reshape(3, 3)
    edges = product(*((-0.5, size - 0.5) for size in oriented.GetSize()))
    world_corners = np.asarray([
        oriented.TransformContinuousIndexToPhysicalPoint(index) for index in edges
    ])
    projected_corners = world_corners @ axes
    lower = projected_corners.min(axis=0)
    upper = projected_corners.max(axis=0)
    spacing = np.asarray(oriented.GetSpacing() if output_spacing is None else output_spacing)
    size = np.ceil((upper - lower) / spacing).astype(int).tolist()
    origin = (axes @ (lower + spacing / 2)).tolist()
    if verbose:
        print(f"Oblique image resampled to {orientation}")
    return sitk.Resample(
        oriented, size, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear,
        origin, spacing.tolist(), desired_direction, float(default_value),
        oriented.GetPixelID(),
    )


def resampleVolume(outspacing, vol,resamplemethod=sitk.sitkNearestNeighbor):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()
 
    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
 
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(resamplemethod)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol

def resample_image(input_image, reference_image,default=0):
    # 创建一个重采样过滤器
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    
    # 设置插值方法，通常使用线性插值
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    
    # 设置输出使用参考图像的spacing, origin和direction
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    
    # 设置输出图像的size
    resampler.SetSize(reference_image.GetSize())
    resampler.SetDefaultPixelValue(default)
    
    # 执行重采样
    return resampler.Execute(input_image)