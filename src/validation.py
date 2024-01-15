import json
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from lib.datasets.dataset import jde
from lib.models.networks.resnet_dcn import BasicBlock
from lib.models.networks.resnet_dcn import PoseResNet
from lib.models.networks.resnet_dcn import get_pose_net
from lib.opts import opts

def run(opt):
    num_layers = [2, 2, 2, 2]
    # num_layers = [3, 4, 6, 3]
    heads = {'scene': 5}
    head_conv = 256

    # model = get_pose_net(num_layers, heads, head_conv)
    model = PoseResNet(BasicBlock, num_layers, heads, head_conv)

    checkpoint = torch.load('/mnt/storage1/FairMOT_auto/models/FairMOT_classhead.pth')
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    
    model.eval()

    f = open(opt.data_cfg)  # '../src/lib/cfg/mot15.json',
    data_config = json.load(f)
    testset_paths = data_config['test']  
    dataset_root = data_config['root']  
    print("Testset root: %s" % dataset_root)
    f.close()

    transforms = T.Compose([T.ToTensor()])
    Dataset = jde.JointDataset
    dataset = Dataset(opt=opt,
                      root=dataset_root,
                      paths=testset_paths,
                      img_size=(1088, 608),
                      augment=False,
                      transforms=transforms)
    
    test_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)
    # print("DATASET", Dataset['input'])
    # print("TEST_LOADER",test_loader)
    correct = 0
    total = 0

    output_dir = 'classification_results'
    os.makedirs(output_dir,exist_ok=True)

    for data in test_loader:
        inputs = data['input']
        # print(type(inputs))
        targets = data['scene']
        targets = targets.argmax(dim=-1, keepdim=True)
        print("targets : ", targets)
        with torch.no_grad():
            outputs = model(inputs)
            # print("model_output:", outputs)
            # print(type(outputs))
            # print(outputs.size())
            # outputs = torch.tensor(outputs)  # 리스트를 텐서로 변환
            output_tensors = [item['scene'] for item in outputs]
            outputs = torch.stack(output_tensors)
            # print("outputs :",outputs)
            predicted = outputs.argmax(dim=-1, keepdim=True)
            # predicted = torch.max(outputs, dim=-1)
            print("predicted :",predicted)
            # total += targets.size(0)
            correct += (predicted == targets).sum().item()
            print(correct)
            # correct += predicted.eq(targets.view_as(predicted)).sum().item()
        # print(inputs.size(0))
        for i in range(inputs.size(0)):
            image = inputs[i].permute(1,2,0).numpy()
            # plt.imshow(image)
            # plt.title(f'True Label: {targets[i]}, Predicted Label: {predicted[i]}')
            # plt.show()
            true_label = targets[i].item()
            predicted_label = predicted[i].item()

            from datetime import datetime
            time_stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            image_filename = f'true_{true_label}_predicted_{predicted_label}_{time_stamp}.png'
            image_path = os.path.join(output_dir, image_filename)
            
            plt.imshow(image)
            plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
            plt.savefig(image_path, format='png')
            plt.close()

            print(f"Saved image to: {image_path}")

    # accuracy = 100. * correct / total
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), accuracy))
    # return accuracy

if __name__ == '__main__':
    opt = opts().parse()
    run(opt)
    # print("Accuracy: {:.2f}%".format(accuracy))
