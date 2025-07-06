from tqdm import tqdm

from random import choice as randomly_choose_from

import torch
import numpy as np

from PIL import Image

from torch import Tensor as Data
from torch.nn import ReLU as ActivationFunction
from torch.nn import Sequential as NeuralNetwork
from torch.nn import Linear as FullyConnectedLayer
from torch.nn import CrossEntropyLoss as LossFunction
from torch.nn import Softmax as ActivationLayerProbabilities

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import MNIST as digits_dataset
from torchvision.transforms import ToTensor

from matplotlib.pyplot import plot as plot_graph
from matplotlib.pyplot import show as show_graph
from matplotlib.pyplot import imshow as show_image

def prepare_data(
    dataset
):
    # Տվյալները բաժանում ենք ուսուցման ու թեստավորման 
    # համար՝ 80% ուսուցման և 20% թեստավորման (սովորաբար այս հարաբերակցությունն է ընդունված)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    return random_split(dataset, [train_size, test_size])

def train_model(
        model,
        train_dataset,
        test_dataset,
        loss_function,
        optimizer,
        batch_size=64,
        epochs=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
    train_accuracy_history = []  # պահելու ենք ուսուցման ճշգրտության պատմությունը
    test_accuracy_history = []  # պահելու ենք թեստավորման ճշգրտության պատմությունը

    # dataloader-ներով ենք «կերակրում» մոդելը սովորելու ընթացքում
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Մոդելը սովորեցնում ենք շրջանների ընթացքում
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Շրջան {epoch+1}/{epochs}", leave=False)
        
        # Մոդելի ուսուցում՝ տվյալների ավելի փոքր խմբերով
        for batch_idx, (images, labels) in progress_bar:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1) # ընթացիկ կանխատեսում
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item() # համեմատում ճշմարտության հետ

            current_acc = correct_train / total_train * 100 # ընթացիկ ճշգրտություն
            progress_bar.set_postfix({
                "Խումբ": batch_idx + 1,
                "Ճշտություն": f"{current_acc:.2f}%"
            })

        train_accuracy = correct_train / total_train * 100
        train_accuracy_history.append(train_accuracy)
        
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = correct_val / total_val * 100
        test_accuracy_history.append(val_accuracy)
        
        print(f"Շրջան {epoch+1}/{epochs} - Սովորելու Ճշտություն՝ {train_accuracy:.2f}% | Վավերացման Ճշտություն՝ {val_accuracy:.2f}%")

    return train_accuracy_history, test_accuracy_history

def test(
    model,
    test_dataset,
    batch_size=64,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    # վերջնական ստուգում
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    return accuracy

def predict(
    model,
    image,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.eval()
    
    if isinstance(image, torch.Tensor):
        # եթե պատկերն արդեն Tensor է, ապա պետք է միայն ձևափոխել չափսն ու հաշվարկային սարքը
        if len(image.shape) == 1:
            # եթե պատկերն արդեն 1D է
            image = image.unsqueeze(0).to(device)
        elif len(image.shape) == 2:
            # խմբի չափողականությունն արդեն ավելացված է
            image = image.view(image.size(0), -1).to(device)
        else:
            # պետք է ձևափոխել պատկերը 1D տեսքի ու ավելացնել խմբի չափ
            image = image.view(1, -1).to(device)
    else:
        # ձևափոխում ենք պատկերը Tensor-ի
        image = torch.tensor(image, dtype=torch.float32).view(1, -1).to(device)
    
    with torch.no_grad():
        output = model(image) # մոդելի կանխատեսում
        _, predicted = torch.max(output.data, 1) # կանխատեսման արդյունքը
    
    return predicted.item() # վերադարձնում ենք կանխատեսված դասը (թվանշանը)

def visualize(
    image
):
    # Նկարի ձևափոխում՝ 28x28 չափսի մոխրագույն պատկեր
    image = image.view(28, 28).cpu().numpy()
    show_image(image, cmap='gray')
    return image

def predict_custom_image(
    image_path,
    predict_function
):
    # Նկարի ներբեռնում՝ որպես մոխրագույն պատկեր (նույնիսկ եթե գունավոր է)
    img = Image.open(image_path).convert('L')
    
    # Նկարը փոքրացնենք 28x28 չափի
    img = img.resize((28, 28))

    # Նկարը վերածում ենք numpy զանգվածի
    img_np = np.array(img)
    
    # Նկարը վերածում ենք PyTorch տ tensor-ի
    img_tensor = torch.tensor(img_np, dtype=torch.float32).view(-1) / 255.0
    
    # Եթե նկարը եռամիաչափ է (գունավոր), ապա պակասացնենք չափողականությունը
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Գուշակում
    predicted = predict_function(img_tensor)
    
    return predicted, img  # վերադարձնում ենք կանխատեսված թվանշանը
    