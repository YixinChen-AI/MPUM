from torch import nn
import numpy as np
import torch
import einops
from categories import prediction as prediction_label_tissue

class Controller(nn.Module):
    def __init__(self, config,in_channel,out_channel,stride=1):
        super().__init__()
        self.config = config
        if config["modelsize"] == "tiny":
            self.latent = 32
        elif config["modelsize"] == "base":
            self.latent = 64
        elif config["modelsize"] == "huge":
            self.latent = 128
            
        self.conv1 = nn.Conv3d(in_channel,self.latent,bias=False,kernel_size=1)
        if config["modality"] == "ct":
            self.fc1 = nn.Linear(config["modalitydimension"], self.latent * 3 * 3 * 3)
        elif config["modality"] == "mr":
            self.fc1 = nn.Linear(config["modalitydimension"], self.latent * 3 * 3 * 3)
        elif config["modality"] == "pet":
            self.fc1 = nn.Linear(config["modalitydimension"], self.latent * 3 * 3 * 3)
        self.conv2 = nn.Conv3d(config["tissuenumber"],out_channel,bias=False,kernel_size=3,padding=1,stride=stride)
        self.in_channel = in_channel
    def forward(self,orix,controllerspace):
        x = self.conv1(orix)
        kernel = self.fc1(controllerspace)
        kernel = einops.rearrange(kernel, 'a (b c d e) -> a b c d e', b=self.latent, c=3,d=3,e=3)
        saliency = nn.functional.conv3d(x,kernel,None,1,1)
        x = self.conv2(saliency)
        return x,saliency
class Block(nn.Module):
    def __init__(self, config,in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.config = config
        self.in_channel = in_channel
        self.out_channel = out_channel
        # branch1
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(out_channel, affine=True,track_running_stats=False)
        self.act1 = nn.LeakyReLU(inplace=True)

        # branch2
        self.tissueTUC = Controller(config,in_channel,config["tissuenumber"],stride=stride)
        # branch3
        # self.lesionTUC = Controller(config,in_channel,config["tissuenumber"],stride=stride)
        self.tail = nn.Conv3d(out_channel+ config["tissuenumber"],out_channel,kernel_size=1,padding=0,stride=1)
        if stride == 2 or (stride==1 and in_channel != out_channel):
            self.res = nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
        else:
            self.res = None
        self.norm2 = nn.InstanceNorm3d(out_channel, affine=True,track_running_stats=False)
        self.act2 = nn.LeakyReLU(inplace=True)
    def forward(self, orix,controllerspace,lastblock=False):
        # branch1
        x1 = self.conv1(orix)
        x1 = self.act1(self.norm1(x1))  
        # branch2
        x2,saliency = self.tissueTUC(orix,controllerspace)
        # branch3
        # x3 = self.lesionTUC(orix)

        if lastblock:
            return None,x2
        x12  = torch.concat([x1,x2],axis=1)
        x = self.tail(x12)

        if self.res:
            x += self.res(orix)
        else:
            x += x
        x = self.act2(self.norm2(x))
        return x,saliency

class MPUM(nn.Module):
    def __init__(self,config,tissueclip, in_channel):
        super().__init__()
        self.input_channels = in_channel
        self._deep_supervision = False
        self.config = config
        self.register_buffer('tissueclip', tissueclip)
        # self.register_buffer('lesionclip', lesionclip)
        self.backprojection_matrix_tissue = nn.Sequential(
            nn.Linear(self.tissueclip.shape[1],1024),
        )
        # self.backprojection_matrix_lesion = nn.Sequential(
        #     nn.Linear(self.tissueclip.shape[1]+self.lesionclip.shape[1],1024),
        # )
        
        
        if config["modelsize"] == "tiny":
            self.channels = [32,64,128,256] 
        elif config["modelsize"] == "base":
            self.channels = [64,128,256,512] 
        elif config["modelsize"] == "huge":
            self.channels = [128,256,512,1024]
            
        self.head = nn.Conv3d(1,self.channels[0],kernel_size=3,padding=1,stride=2)
        self.Block1_2 = Block(config,self.channels[0],self.channels[0])
        self.Block2_1 = Block(config,self.channels[0],self.channels[1],stride=2)
        self.Block2_2 = Block(config,self.channels[1],self.channels[1] )
        self.Block3_1 = Block(config,self.channels[1],self.channels[2] ,stride=2)
        self.Block3_2 = Block(config,self.channels[2],self.channels[2] )
        self.Block4_1 = Block(config,self.channels[2],self.channels[3] ,stride=2)
        self.Block4_2 = Block(config,self.channels[3],self.channels[3])
        # self.Block5_1 = Block(self.channels[3],self.channels[4] ,stride=2)
        # self.Block5_2 = Block(self.channels[4],self.channels[4] )
        self.upsample = nn.Upsample(scale_factor=2)
        # self.Block4_3 = Block(self.channels[4] + self.channels[3],self.channels[3] )
        # self.Block4_4 = Block(self.channels[3],self.channels[3] )
        self.Block3_3 = Block(config,self.channels[3] + self.channels[2],self.channels[2] )
        self.Block3_4 = Block(config,self.channels[2],self.channels[2])
        self.Block2_3 = Block(config,self.channels[2] + self.channels[1],self.channels[1] )
        self.Block2_4 = Block(config,self.channels[1],self.channels[1])  
        self.Block1_3 = Block(config,self.channels[1] + self.channels[0],self.channels[0] )
        self.Block1_4 = Block(config,self.channels[0],self.channels[0])  

        self.ct_projection_matrix = nn.Linear(1024,config["modalitydimension"],bias=False)
        self.pet_projection_matrix = nn.Linear(1024,config["modalitydimension"],bias=False)
        self.mr_projection_matrix = nn.Linear(1024,config["modalitydimension"],bias=False)
        self.saliency=False
    def forward(self, x):
        controllerspace = self.backprojection_matrix_tissue(self.tissueclip)
        
        if self.config["modality"] == "ct":
            controllerspace = self.ct_projection_matrix(controllerspace)
        elif self.config["modality"] == "pet":
            controllerspace = self.pet_projection_matrix(controllerspace)
        elif self.config["modality"] == "mr":
            controllerspace = self.mr_projection_matrix(controllerspace)
        # deep_supervision = []
        x1_1 = self.head(x)
        x1_2,o7 = self.Block1_2(x1_1,controllerspace)
        x2_1,_ = self.Block2_1(x1_2,controllerspace)
        x2_2,o6 = self.Block2_2(x2_1,controllerspace)
        x3_1,_ = self.Block3_1(x2_2,controllerspace)
        x3_2,o5 = self.Block3_2(x3_1,controllerspace)
        x4_1,_ = self.Block4_1(x3_2,controllerspace)
        x4_2,o4 = self.Block4_2(x4_1,controllerspace)
        # x5_1,_ = self.Block5_1(x4_2,ctembedding)
        # x5_2,o5 = self.Block5_2(x5_1,ctembedding)

        # x4_3,_ = self.Block4_3(torch.concat([self.upsample(x5_2),x4_2],dim=1),ctembedding)
        # x4_4,o4 = self.Block4_4(x4_3,ctembedding)
        x3_3,_ = self.Block3_3(torch.concat([self.upsample(x4_2),x3_2],dim=1),controllerspace)
        x3_4,o3 = self.Block3_4(x3_3,controllerspace)
        x2_3,_ = self.Block2_3(torch.concat([self.upsample(x3_4),x2_2],dim=1),controllerspace)
        x2_4,o2 = self.Block2_4(x2_3,controllerspace)
        x1_3,_ = self.Block1_3(torch.concat([self.upsample(x2_4),x1_2],dim=1),controllerspace)
        _,o1 = self.Block1_4(self.upsample(x1_3),controllerspace,lastblock=True)

        # print(self.saliency)
        # if self.saliency:
        return o1,[o1,o2,o3,o4]
        # else:
        #     return o1
        # if self._deep_supervision:
        #     return o1,[o2,o3,o4,o5,o6,o7]
        # else:
        #     return o1
    # def load_model(self, checkpoint):
    #     # 获取当前模型的状态字典
    #     current_state = self.state_dict()
    #     # 为每个不匹配的权重进行切片操作
    #     for key, param in checkpoint.items():
    #         if key in current_state:
    #             if param.size() != current_state[key].size():
    #                 # 进行切片以匹配模型中的相应参数形状
    #                 slices = tuple(slice(0, min(param.size(i), current_state[key].size(i))) for i in range(len(param.size())))
    #                 param = param[slices]
    #                 print(f"Adjusted {key} from {checkpoint[key].size()} to {param.size()}")
    #             current_state[key] = param
    #         else:
    #             print(f"Skipping {key} because it's not in the current model.")
    #     self.load_state_dict(current_state,strict=False)
    
    def get_unispace(self):
        return self.backprojection_matrix_tissue(self.tissueclip)
    def get_ctspace(self):
        controllerspace = self.backprojection_matrix_tissue(self.tissueclip)
        controllerspace = self.ct_projection_matrix(controllerspace)
        return controllerspace
    def get_mrspace(self):
        controllerspace = self.backprojection_matrix_tissue(self.tissueclip)
        controllerspace = self.mr_projection_matrix(controllerspace)
        return controllerspace
    def get_petspace(self):
        controllerspace = self.backprojection_matrix_tissue(self.tissueclip)
        controllerspace = self.pet_projection_matrix(controllerspace)
        return controllerspace
    def get_label(self):
        return prediction_label_tissue

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 输入维度：(batch_size, 133, 128, 128, 128)
        self.features = nn.Sequential(
            # 第一层减少空间维度
            nn.Conv3d(133, 64, kernel_size=3, stride=2, padding=1), # 输出：(batch_size, 64, 64, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            
            # 进一步降低维度和增加通道数
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # 输出：(batch_size, 128, 32, 32, 32)
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 继续减小维度
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1), # 输出：(batch_size, 256, 16, 16, 16)
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最后一层，转换为更小的空间尺寸
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1), # 输出：(batch_size, 512, 8, 8, 8)
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=0),  # 输出：(batch_size, 1, 1, 1, 1)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    