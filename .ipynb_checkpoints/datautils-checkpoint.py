import SimpleITK as sitk


def adjust_image_direction(image, desired_direction,verbose=True):
    """
    调整图像的方向到所需方向，如果需要的话会翻转图像。
    
    :param image: SimpleITK图像对象
    :param desired_direction: 一个包含九个元素的元组，表示所需的图像方向
    :return: 调整方向后的图像
    """
    # 获取当前图像的方向
    current_direction = image.GetDirection()
    if len(current_direction) == 16:
        input_array = np.array(current_direction)
        matrix_4x4 = input_array.reshape((4, 4))
        sub_matrix_3x3 = matrix_4x4[:3, :3]
        current_directiony = sub_matrix_3x3.flatten()

    # 计算需要翻转的轴
    flip_axes = [current_direction[i] * desired_direction[i] < 0 for i in [0,4,8]]
    if verbose:
        print("the axis need to be flip?:", flip_axes)
    # 翻转图像
    adjusted_image = sitk.Flip(image, flip_axes)

    # 设置新的方向
    adjusted_image.SetDirection(desired_direction)

    return adjusted_image
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