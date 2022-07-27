import os
import math
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import torch.optim.lr_scheduler as lr_scheduler
import time
from model import efficientnetv2_s as create_model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "test": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), '../'))
    image_path = data_root + '/laji_data/'

    train_dataset = datasets.ImageFolder(root=image_path + 'train',
                                         transform=data_transform['train'])

    flower_list = train_dataset.class_to_idx  # 获取训练数据集中类别名称及所对应的索引

    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_dataset = datasets.ImageFolder(root=image_path + 'test',
                                       transform=data_transform['test'])
    val_num = len(val_dataset)
    batch_size = args.batch_size
    nw = 0  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               # collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             # collate_fn=val_dataset.collate_fn
                                             )


    model = create_model(num_classes=args.num_classes).to(device)  #实例化模型

    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  #学习率
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_function = torch.nn.CrossEntropyLoss()
    # for epoch in range(args.epochs):
    #     # train
    #     train_loss, train_acc = train_one_epoch(model=model,
    #                                             optimizer=optimizer,
    #                                             data_loader=train_loader,
    #                                             device=device,
    #                                             epoch=epoch)
    #
    #     scheduler.step()
    #
    #     # validate
    #     val_loss, val_acc = evaluate(model=model,
    #                                  data_loader=val_loader,
    #                                  device=device,
    #                                  epoch=epoch)
    #
    #     tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    #     tb_writer.add_scalar(tags[0], train_loss, epoch)
    #     tb_writer.add_scalar(tags[1], train_acc, epoch)
    #     tb_writer.add_scalar(tags[2], val_loss, epoch)
    #     tb_writer.add_scalar(tags[3], val_acc, epoch)
    #     tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    for epoch in range(args.epochs):
        model.train()  # 训练中 使用dropout方法
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))

            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

            rate = (step + 1) / len(train_loader)
            a = '*' * int(rate * 50)
            b = '.' * int((1 - rate) * 50)
            print('\rtrain loss:{:^3.0f}%[{}-->{}]{:.3f}'.format(int(rate * 100), a, b, running_loss / (step + 1)),
                  end='')
        print()
        print(time.perf_counter() - t1)

        model.eval()  # 测试过程不使用dropout方法
        acc = 0.0
        with torch.no_grad():
            for data_test in val_loader:
                test_images, test_labels = data_test
                outputs = model(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                # torch.max 返回最大值和索引
                acc += (predict_y == test_labels.to(device)).sum().item()
            accurate_test = acc / val_num
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
            print('[epoch %d] train_loss:%.3f test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, acc / val_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    #
    # # 数据集所在根目录
    # # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")

    # download model weights
    # 链接: https://pan.baidu.com/s/1uZX36rvrfEss-JGj4yfzbQ  密码: 5gu1
    parser.add_argument('--weights', type=str, default='./torch_efficientnetv2/pre_efficientnetv2-s.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)