import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torchvision.models import resnet50,resnet34,resnet18
from PIL import Image
from sklearn import metrics

# device_0 = torch.device("cuda:0")
# device_1 = torch.device("cuda:1")
# file_train = '/home/wjp21/project1/src/HiFaceGAN/results/hf_ours_split_final_cfee'
# file_test = '/home/wjp21/project1/stronger/pix2pix_ours/data/Ours/test_cls'
# pth_name = 'ours_hif_cfee_3'
parser = argparse.ArgumentParser(description='eval.py')

parser.add_argument('--file_train',help='input your train file here')
parser.add_argument('--file_test',help='input your test file here',default='/home/wjp21/project1/stronger/resnet_benchmark/data/Ours/test_cls')
parser.add_argument('--pth_name',help='input your pth name here',default='default')
parser.add_argument('--device_num',default=0)

args = parser.parse_args()
file_train = args.file_train
file_test  =args.file_test
pth_name = args.pth_name
device_num = torch.device("cuda:"+str(args.device_num))

def train_resnet(file,pth_name,device,train_num_2):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                    #进行标准化的方法的参数是参考官网
        "val": transforms.Compose([transforms.Resize(256),#先通过resize将最小遍缩放到256，
                                   transforms.CenterCrop(224),#在使用中心裁剪
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # file = './data/Oulu/train/nir_cls' #'data/Ours/train/nir_cls'
    # pth_name = 'oulu_nir_baseline'#'ours_nir_baseline'
    train_dataset = datasets.ImageFolder(root=file,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
 
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('/home/wjp21/project1/eval/json/'+pth_name+'_class_indices.json', 'w') as json_file:
        json_file.write(json_str)
 
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    net = resnet50() 
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 6)
    net.to(device)
 
    # define loss function
    loss_function = nn.CrossEntropyLoss()
 
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
 
    epochs = 200
    best_acc = 0.0
    save_path = '/home/wjp21/project1/eval/weight/'+pth_name+'_'+str(train_num_2)+'.pth' #保存权重的名字也进行相应的修改
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()#重要的
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
 
            # print statistics
            running_loss += loss.item()
 
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     running_loss / (step + 1))
        lr_scheduler.step()
        torch.save(net.state_dict(), save_path)
 
    print('Finished Training')
 

for i in range(1,4):
    train_resnet(file=file_train,pth_name=pth_name,device=device_num,train_num_2=i)

Acc=[]
f1=[]
Recall=[]
Prec=[]
def test(file,pth_name,device,test_num):
    os.system("touch /home/wjp21/project1/eval/result/"+pth_name+".txt")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # file='./data/Oulu/test/nir_cls'
    test_dataset = datasets.ImageFolder(root=file,
                                         transform=data_transform)
    # [N, C, H, W]
    # read class_indict
    json_path = '/home/wjp21/project1/eval/json/'+pth_name+'_class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # create model 使用哪个网络就传入哪个网络
    model = resnet50()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 6) #将最后一个新连接层，替换成自己的新建的一个全连接层，5表示要分类的类别个数
    # load model weights
    weights_path = '/home/wjp21/project1/eval/weight/'+pth_name+'_'+str(test_num)+'.pth'  #权重进行相应的改变
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    # prediction
    model.eval()
    with torch.no_grad():#不对损失梯度进行跟踪
        # predict class
        predlist=[]
        labellist=[]
        for step, data in enumerate(test_dataset):
            images, labels = data
            images=torch.unsqueeze(images,0)
            output=model(images.to(device))
            pred=torch.argmax(torch.softmax(torch.squeeze(output),0),0).item()
            predlist.append(pred)
            labellist.append(labels)
    acc=metrics.accuracy_score(labellist, predlist)
    print("round:"+test_num)
    print(acc)
    labels=[0,1,2,3,4,5]
    F1=metrics.f1_score(labellist,predlist,labels=labels,average='macro')
    recall=metrics.recall_score(labellist,predlist,labels=labels,average='macro')
    prec=metrics.precision_score(labellist,predlist,labels=labels,average='macro')
    print(F1)
    print(recall)
    print(prec)
    with open("/home/wjp21/project1/eval/result/"+pth_name+".txt",'a') as file:
        file.write('round'+test_num+'\nacc:'+acc+'\nF1:'+F1+'\nrecall:'+recall+'\nprec:'+prec+'\n')
    Acc.append(acc)
    f1.append(F1)
    Recall.append(recall)
    Prec.append(prec)
with open("/home/wjp21/project1/eval/result/"+pth_name+".txt",'a') as file:
    file.write('------------------------'+pth_name+'-------------------------------')
for j in range(1,4):
    test(file_test,pth_name,device = device_num,test_num=j)

with open("/home/wjp21/project1/eval/result/"+pth_name+".txt",'a') as file:
    file.write('----------Result----------\n'+'acc:'+sum(Acc)/len(Acc)+'\nF1:'+sum(f1)/len(f1)+'\nrecall:'+sum(Recall)/len(Recall)+'\nprec:'+sum(Prec)/len(Prec))

    

