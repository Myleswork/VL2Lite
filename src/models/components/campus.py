import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

import open_clip

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)

class TeacherStudent(nn.Module):
    def __init__(self, teacher, student, data_attributes, use_teacher=True):
        super(TeacherStudent, self).__init__()
        self.teacher, self.align, self.frozen_nlp_features = None, None, None
        self.data_attributes = data_attributes
        self.student = StudentNet(student, data_attributes.class_num, use_teacher)
        if use_teacher:
            device = next(self.student.model.resnet_features.parameters()).device
            self.teacher = TeacherNet(teacher)
            self.align = AlignNet(self.teacher.last_features_dim, self.student.num_features)
            self.frozen_nlp_features = self.get_frozen_nlp_features(data_attributes)
    
    def get_frozen_nlp_features(self, attributes):
        prompt_tmpl = attributes.prompt_tmpl
        classes_list = list(attributes.classes.values())
        text_tokens = self.teacher.tokenizer([prompt_tmpl.format(word) for word in classes_list])
        nlp_features = self.teacher.encode_text(text_tokens).detach()
        return feature_norm(nlp_features)
    
    def forward(self, x):
        if self.teacher:
            #教师特征不变
            clip_img_features = self.teacher(x)
            frozen_nlp_features = self.frozen_nlp_features.to(clip_img_features.device)
            #学生特征需要返回值
            #spatial_features = (B, C_student, H, W)
            #hidden_features = (B, C_student)
            #out = (B, K_classes)
            spatial_features, hidden_features, out = self.student(x)
            #投影层
            # aligned_img, aligned_nlp = self.align(clip_img_features, frozen_nlp_features)
            # hidden_features, out = self.student(x)
            #return hidden_features, out, clip_img_features, frozen_nlp_features, aligned_img, aligned_nlp
            aligned_nlp = self.align(clip_img_features, frozen_nlp_features)
            aligned_img = None
            return (
                spatial_features,                      #[0] (B, C_student, H, W)  学生空间特征
                self.student.model.linear_cls.weight,  #[1](K, C_s)  学生分类器权重
                hidden_features,                       #[2](B, C_student) 学生全局特征
                out,                                   #[3](B, K_classes) 学生分类输出logits，
                clip_img_features,                     #[4](B, C_teacher) 教师图像特征
                frozen_nlp_features,                   #[5](K, C_teacher) 教师文本特征
                aligned_img,                           #[6](B, C_student) 对齐后的图像特征，原L_vis目标
                aligned_nlp                            #[7](K, C_student) 对齐后的文本特征，原L_text目标,CATKD教师权重
            )
        #非蒸馏模式
        out_tuple = self.student(x)
        if self.student.use_teacher:
            return out_tuple[1], out_tuple[2]
        else:
            return out_tuple[-1]


class TeacherNet(nn.Module):
    def __init__(self, teacher):
        super(TeacherNet, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(teacher.arch, pretrained=teacher.pretrained)
        self.model.requires_grad_(False)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(teacher.arch)
        self.last_features_dim = self.model.transformer.resblocks[-1].mlp.c_proj.out_features

    def encode_image(self, x):
        return self.model.encode_image(x)

    def encode_text(self, x):
        return self.model.encode_text(x)

    def forward(self, x):
        with torch.no_grad():
            clip_img_features = self.encode_image(x).detach()
        clip_img_features = feature_norm(clip_img_features)
        return clip_img_features
    

class AlignNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(AlignNet, self).__init__()

        # self.align_img_layer = nn.Sequential(
        #     nn.Linear(in_features, out_features), 
        #     nn.ReLU(), 
        #     nn.Linear(out_features, out_features)
        #     )
        self.align_nlp_layer = nn.Sequential(
            nn.Linear(in_features, out_features), 
            nn.ReLU(), 
            nn.Linear(out_features, out_features)
        )
    
    def forward(self, x, clip_nlp_features):
        # align_img = self.align_img_layer(x)
        align_nlp = self.align_nlp_layer(clip_nlp_features)
        # return feature_norm(align_img), feature_norm(align_nlp)
        return feature_norm(align_nlp)



class StudentNet(nn.Module):
    def __init__(self, student, class_num, use_teacher=True):
        super(StudentNet, self).__init__()
        self.use_teacher = use_teacher
        self.num_features = None
        if self.use_teacher:
            origin_model = models.__dict__[student.arch](pretrained=True)
            self.model  = ModifiedResNet(origin_model, class_num)
            self.num_features = self.model.num_features
        else:
            self.model = models.__dict__[student.arch](pretrained=True)
            try:
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, class_num)
            except:
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, class_num)

    def forward(self, x):
        # if self.use_teacher:
        #     hidden_features, out = self.model(x)
        #     return feature_norm(hidden_features), out
        # out = self.model(x)
        # return out
        if self.use_teacher:
            return self.model(x)
        out = self.model(x)
        return (None, None, out)


class ModifiedResNet(torch.nn.Module):
    def __init__(self, origin_model, classnum):
        super(ModifiedResNet, self).__init__()
        # self.resnet = origin_model
        
        try:
            num_features = origin_model.fc.in_features
            # self.resnet.fc = nn.Identity()
            self.resnet_features = nn.Sequential(*list(origin_model.children())[:-2])
            self.resnet_pool = origin_model.avgpool
            self.num_features = num_features
        except Exception as e:
            # num_features = origin_model.classifier[1].in_features
            # self.resnet.classifier  = nn.Identity()   
            print(f"Warning: ResNet parsing failed ({e}). Falling back to classifier parsing.")
            num_features = origin_model.classifier[1].in_features
            self.resnet_features = origin_model.features
            self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1)) # 假设
            self.num_features = num_features        
        self.linear_cls = nn.Linear(num_features, classnum)
        # self.num_features = num_features

    def forward(self, x):
        # hidden_features = self.resnet(x)
        # out = self.linear_cls(hidden_features)
        #得到空间特征图(B, C, H, W)
        spatial_features = self.resnet_features(x)
        #得到全局特征(B, C)
        hidden_features = self.resnet_pool(spatial_features)
        hidden_features = torch.flatten(hidden_features, 1)
        #得到logits(B, K)
        out = self.linear_cls(hidden_features)


        return spatial_features, hidden_features, out





if __name__ == "__main__":
    _ = TeacherStudent()
