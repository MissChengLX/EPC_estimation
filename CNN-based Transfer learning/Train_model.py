import torch.nn as nn
import torch
from torch.autograd import Variable
import pandas as pd
from torch.optim import lr_scheduler
import torchvision
import time
import numpy as np
from torchsummary import summary
import copy
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torchvision

def plot_training(acc_hist,name):
    train_loss=[]
    val_loss = []
    train_acc = []
    val_acc = []
    for hist in acc_hist[::2]:
        train_loss.append(hist[0])
        train_acc.append(hist[1])
    for hist in acc_hist[1::2]:
        val_loss.append(hist[0])
        val_acc.append(hist[1])
    data = pd.DataFrame(data=train_acc,columns=['train_acc'])
    data['val_acc'] = val_acc
    data['train_loss'] = train_loss
    data['val_loss'] = val_loss
    data.to_csv('./'+name+'.csv',index=False)
    print('----loss_acc save!-----')

    plt.figure(12)
    plt.subplot(121)
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'b', label='train_accuracy')
    plt.plot(epochs, val_acc, 'r', label='test_accuracy')
    plt.title('Train and Val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(122)

    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label='train_loss')
    plt.plot(epochs, val_loss, 'r', label='test_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test loss')
    plt.legend()
    plt.savefig('./'+name+'.png',dpi=300)


def finetune(name_,epochs_n,model, dataloaders, optimizer, criterion, use_lr_schedule=False):

    N_EPOCH = epochs_n
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()
    best_acc = 0.0
    acc_hist = []
    hist_loss =[]
    hist_acc=[]

    for epoch in range(1, N_EPOCH + 1):
        if use_lr_schedule:
            lr_schedule(optimizer, epoch)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct.double() / len(dataloaders[phase].dataset)
            acc_hist.append([epoch_loss, epoch_acc.item()])
            hist_loss.append(epoch_loss)
            hist_acc.append(epoch_acc.item())
            print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, N_EPOCH, phase, epoch_loss,
                                                                                  epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './models_results/best_{}_{}-{}.pth'.format('ResNet50', 'Google', epoch))
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_pass // 60, time_pass % 60))
    print('------Best acc: {}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    #torch.save(model.state_dict(), best_model_path)
    name_= 'best_model_'+name_+'.pth'
    #torch.save(model, './Best_models/best_model_Google_vgg16_1.pth')
    torch.save(model, './Best_models/'+name_)
    #print('Best model saved!')
    #da = pd.Dataframe(data=acc_hist)
    plot_training(acc_hist,name_)
    return model 

