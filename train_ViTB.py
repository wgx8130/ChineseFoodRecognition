import os
import math
import argparse
import json
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from timm.models.vision_transformer import vit_base_patch16_224_in21k
from utils import read_split_data, train_one_epoch, evaluate
#from torchvision import transforms
#
#
#from my_dataset import MyDataSet
#from vit_pytorch import ViT

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()



    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
    #    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    image_path = args.data_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 实例化训练数据集
#    train_dataset = MyDataSet(images_path=train_images_path,
#                              images_class=train_images_label,
#                              transform=data_transform["train"])
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    
    food_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in food_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    # 实例化验证数据集
#    val_dataset = MyDataSet(images_path=val_images_path,
#                            images_class=val_images_label,
#                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    model = vit_base_patch16_224_in21k(pretrained=True, num_classes=208)

##    if args.weights != "":
##        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
##        weights_dict = torch.load(args.weights, map_location=device)
#        # 删除不需要的权重
#    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
#        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
#    for k in del_keys:
#        del weights_dict[k]
#    #print(model.load_state_dict(weights_dict, strict=False))
    
    if args.freeze_layers:
        #print('**********')
        for name, para in model.named_parameters():
            #print(model.named_parameters())
            #print(name)
            #print(para)
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                #print('**********')
                #print(para)
                print("training {}".format(name))

    model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
#    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.SGD(pg, lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
#    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_acc = 0.0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc, val_acc_top5 = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "val_acc_top5", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], val_acc_top5, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

        #torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(epoch)
            torch.save(model.state_dict(), "./weights/model-best.pth")
    print('The best val_accuracy: %.4f' %
              (best_acc))
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=208)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=64)
#    parser.add_argument('--lr', type=float, default=0.001)
#    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="your path") # change it to your path
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)

    opt = parser.parse_args()

    main(opt)
