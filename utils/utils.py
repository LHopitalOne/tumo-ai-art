import io
import requests

from tqdm import tqdm

from random import choice as randomly_choose_from

import torch
import numpy as np

from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

from torch import Tensor as Data
from torch.nn.functional import softmax as probabilities_function
from torch.nn import ReLU as ActivationFunction
from torch.nn import Sequential as NeuralNetwork
from torch.nn import Linear as FullyConnectedLayer
from torch.nn import CrossEntropyLoss as LossFunction
from torch.nn import Softmax as ActivationLayerProbabilities

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import MNIST as digits_dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize

from matplotlib.pyplot import plot as plot_graph
from matplotlib.pyplot import show as show_graph
from matplotlib.pyplot import imshow as show_image

def prepare_data(dataset, train_ratio=0.8):
    """
    Տվյալները բաժանում ենք ուսուցման ու թեստավորման համար
    """
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def get_device():
    """Ամենալավ հասանելի սարքը ճանաչել"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def move_to_device(model, device=None):
    """Մոդելը տեղափոխել նշված սարքի վրա"""
    if device is None:
        device = get_device()
    return model.to(device), device

def prepare_batch(images, labels, device, model_type='cnn'):
    """
    Խմբերը պատրաստել տարբեր մոդելների համար
    
    Args:
        images: Մուտքային նկարներ
        labels: Նպատակային պիտակներ
        device: Սարքը տվյալները տեղափոխելու համար
        model_type: 'cnn' կոնվոլյուցիոն համար, 'mlp' ամբողջությամب կապված համար, 'vgg16' VGG16 համար
    """
    # Move to device first
    images = images.to(device)
    labels = labels.to(device)
    
    if model_type == 'mlp':
        # Հարթեցնել ամբողջությամբ կապված մոդելների համար
        # For MLP, flatten the images from (batch_size, channels, height, width) to (batch_size, 784)
        # MNIST images are 28x28=784 pixels
        batch_size = images.size(0)
        images = images.view(batch_size, -1)  # This will flatten to (batch_size, 784)
    elif model_type == 'vgg16':
        # VGG16 համար ապահովում ենք 224x224 չափը
        if images.size(2) != 224 or images.size(3) != 224:
            # Resize to 224x224 if not already
            import torch.nn.functional as F
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    
    return images, labels

def create_vgg16_transforms(input_size=(28, 28), target_size=(224, 224)):
    """
    VGG16 համար անհրաժեշտ transform-ներ ստեղծել
    
    Args:
        input_size: Մուտքային նկարի չափը
        target_size: Նպատակային չափը VGG16 համար
    
    Returns:
        Transform pipeline
    """
    transforms = []
    
    # Resize from input_size to target_size
    if input_size != target_size:
        transforms.append(Resize(target_size))
    
    # Convert to tensor
    transforms.append(ToTensor())
    
    # Normalize for better training (optional but recommended)
    # Using ImageNet normalization values adapted for grayscale
    transforms.append(Normalize(mean=[0.485], std=[0.229]))
    
    return Compose(transforms)

def train_model(
    model,
    train_dataset,
    test_dataset,
    loss_function,
    optimizer,
    batch_size=64,
    epochs=10,
    device=None,
    model_type='cnn',
    print_every=1
):
    """
    Մոդել սովորեցնել ճկուն սարքային աջակցությամբ
    """
    # Սարքի կարգավորում
    if device is None:
        device = get_device()
    
    model, device = move_to_device(model, device)
    print(f"Սովորեցնում ենք սարքի վրա: {device}")
    
    train_accuracy_history = []  # պահելու ենք ուսուցման ճշգրտության պատմությունը
    test_accuracy_history = []  # պահելու ենք թեստավորման ճշգրտության պատմությունը
    
    # dataloader-ներով ենք «կերակրում» մոդելը սովորելու ընթացքում
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Մոդելը սովորեցնում ենք շրջանների ընթացքում
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                          desc=f"Շրջան {epoch+1}/{epochs}", leave=False)
        
        # Մոդելի ուսուցում՝ տվյալների ավելի փոքր խմբերով
        for batch_idx, (images, labels) in progress_bar:
            images, labels = prepare_batch(images, labels, device, model_type)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # ընթացիկ կանխատեսում
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()  # համեմատում ճշմարտության հետ
            
            current_acc = correct_train / total_train * 100  # ընթացիկ ճշգրտություն
            progress_bar.set_postfix({
                "Խումբ": batch_idx + 1,
                "Ճշտություն": f"{current_acc:.2f}%",
            })
        
        train_accuracy = correct_train / total_train * 100
        train_accuracy_history.append(train_accuracy)
        
        # Վավերացման գնահատում
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = prepare_batch(images, labels, device, model_type)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = correct_val / total_val * 100
        test_accuracy_history.append(val_accuracy)
        
        if (epoch + 1) % print_every == 0:
            print(f"Շրջան {epoch+1}/{epochs} - Սովորելու Ճշտություն՝ {train_accuracy:.2f}% | "
                  f"Վավերացման Ճշտություն՝ {val_accuracy:.2f}%")
    
    return train_accuracy_history, test_accuracy_history

def test(
    model,
    test_dataset,
    batch_size=64,
    device=None,
    model_type='cnn'
):
    """
    Մոդելի վերջնական ստուգում
    """
    # Սարքի կարգավորում
    if device is None:
        device = get_device()
    
    model, device = move_to_device(model, device)
    
    # վերջնական ստուգում
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = prepare_batch(images, labels, device, model_type)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    print(f"Վերջնական ճշգրտություն: {accuracy:.2f}%")
    return accuracy

def predict(
    model,
    image,
    device=None,
    model_type='cnn'
):
    """
    Մոդելի կանխատեսում տրված պատկերի համար
    
    Args:
        model: Մոդել
        image: Պատկեր (կարող է լինել tensor, numpy array, կամ list)
        device: Սարք (ավտոմատ որոշվում է եթե None)
        model_type: Մոդելի տեսակ ('cnn', 'mlp', 'vgg16')
    
    Returns:
        Կանխատեսված դասը
    """
    # Սարքի կարգավորում
    if device is None:
        device = get_device()
    
    model, device = move_to_device(model, device)
    model.eval()
    
    if isinstance(image, torch.Tensor):
        # եթե պատկերն արդեն Tensor է
        image_tensor = image.clone()
        
        if model_type == 'vgg16':
            # VGG16 համար ապահովում ենք ճիշտ ձևաչափը
            if len(image_tensor.shape) == 1:
                # 1D -> վերաձևավորում (1, 1, 28, 28) ենթադրելով 28x28 պատկեր
                size = int(image_tensor.numel() ** 0.5)
                image_tensor = image_tensor.view(1, 1, size, size)
            elif len(image_tensor.shape) == 2:
                # 2D -> ավելացնել խմբի և ալիքի չափերը
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            elif len(image_tensor.shape) == 3:
                # 3D -> ավելացնել խմբի չափ
                if image_tensor.size(0) != 1:
                    image_tensor = image_tensor.unsqueeze(0)
            
            # Resize to 224x224 for VGG16
            if image_tensor.size(2) != 224 or image_tensor.size(3) != 224:
                import torch.nn.functional as F
                image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                
        elif model_type == 'cnn':
            # CNN համար պետք է ճիշտ ձևաչափ
            if len(image_tensor.shape) == 1:
                # 1D -> վերաձևավորում (1, 1, 28, 28) ենթադրելով 28x28 պատկեր
                size = int(image_tensor.numel() ** 0.5)
                image_tensor = image_tensor.view(1, 1, size, size)
            elif len(image_tensor.shape) == 2:
                # 2D -> ավելացնել խմբի և ալիքի չափերը
                if image_tensor.size(0) == 1:
                    # Արդեն ունի խմբի չափ
                    size = int(image_tensor.size(1) ** 0.5)
                    image_tensor = image_tensor.view(1, 1, size, size)
                else:
                    # Չունի խմբի չափ
                    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            elif len(image_tensor.shape) == 3:
                # 3D -> ավելացնել խմբի չափ, եթե անհրաժեշտ է
                if image_tensor.size(0) != 1:
                    image_tensor = image_tensor.unsqueeze(0)
        else:  # MLP
            # MLP համար հարթեցնել
            if len(image_tensor.shape) == 1:
                image_tensor = image_tensor.unsqueeze(0)
            else:
                image_tensor = image_tensor.view(1, -1)
                
        image_tensor = image_tensor.to(device)
    else:
        # ձևափոխում ենք պատկերը Tensor-ի
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        if model_type == 'vgg16':
            # VGG16 համար
            if len(image_tensor.shape) == 1:
                size = int(image_tensor.numel() ** 0.5)
                image_tensor = image_tensor.view(1, 1, size, size)
            else:
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            
            # Resize to 224x224
            if image_tensor.size(2) != 224 or image_tensor.size(3) != 224:
                import torch.nn.functional as F
                image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                
        elif model_type == 'cnn':
            # CNN համար
            if len(image_tensor.shape) == 1:
                size = int(image_tensor.numel() ** 0.5)
                image_tensor = image_tensor.view(1, 1, size, size)
            else:
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        else:  # MLP
            # MLP համար հարթեցնել
            image_tensor = image_tensor.view(1, -1)
            
        image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)  # մոդելի կանխատեսում
        _, predicted = torch.max(output.data, 1)  # կանխատեսման արդյունքը
    
    return predicted.item()  # վերադարձնում ենք կանխատեսված դասը (թվանշանը)

def predict_custom_image(
    image_path,
    model,
    device=None,
    model_type='cnn',
    image_size=(28, 28)
):
    """
    Անհատական նկարի կանխատեսում
    
    Args:
        image_path: Նկարի ճանապահը
        model: Մոդել
        device: Սարք (ավտոմատ որոշվում է եթե None)
        model_type: Մոդելի տեսակ ('cnn', 'mlp', 'vgg16')
        image_size: Նկարի չափը (լռությամբ 28x28, բայց VGG16 համար կօգտագործվի 224x224)
    
    Returns:
        Կանխատեսված դասը
    """
    from PIL import Image
    import numpy as np
    
    # Սարքի կարգավորում
    if device is None:
        device = get_device()
    
    model, device = move_to_device(model, device)
    
    # Նկարի ներբեռնում՝ որպես մոխրագույն պատկեր (նույնիսկ եթե գունավոր է)
    img = Image.open(image_path).convert('L')
    
    # VGG16 համար օգտագործում ենք 224x224 չափը
    if model_type == 'vgg16':
        target_size = (224, 224)
    else:
        target_size = image_size
    
    # Նկարը փոքրացնենք նշված չափի
    img = img.resize(target_size)
    
    # Նկարը վերածում ենք numpy զանգվածի
    img_np = np.array(img)
    
    # Նկարը վերածում ենք PyTorch tensor-ի
    img_tensor = torch.tensor(img_np, dtype=torch.float32) / 255.0
    
    # Ձևափոխում ըստ մոդելի տեսակի
    if model_type == 'vgg16':
        # VGG16 համար: (1, 1, 224, 224) -> (1, 3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif model_type == 'cnn':
        # CNN համար: (1, 1, height, width)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    else:  # MLP համար
        # MLP համար: հարթեցնել
        img_tensor = img_tensor.view(1, -1)
    
    img_tensor = img_tensor.to(device)
    
    # Գուշակում
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()  # վերադարձնում ենք կանխատեսված թվանշանը

def predict_custom_image_proba(
    image_path,
    model,
    device=None,
    model_type='cnn',
    image_size=(28, 28)
):
    """
    Անհատական նկարի կանխատեսում հավանականությունների հետ
    
    Args:
        image_path: Նկարի ճանապահը
        model: Մոդել
        device: Սարք (ավտոմատ որոշվում է եթե None)
        model_type: Մոդելի տեսակ ('cnn', 'mlp', 'vgg16')
        image_size: Նկարի չափը (լռությամբ 28x28, բայց VGG16 համար կօգտագործվի 224x224)
    
    Returns:
        Կանխատեսված դասը և հավանականությունը
    """
    from PIL import Image
    import numpy as np
    import torch.nn.functional as F
    
    # Սարքի կարգավորում
    if device is None:
        device = get_device()
    
    model, device = move_to_device(model, device)
    
    # Նկարի ներբեռնում՝ որպես մոխրագույն պատկեր (նույնիսկ եթե գունավոր է)
    img = Image.open(image_path).convert('L')
    
    # VGG16 համար օգտագործում ենք 224x224 չափը
    if model_type == 'vgg16':
        target_size = (224, 224)
    else:
        target_size = image_size
    
    # Նկարը փոքրացնենք նշված չափի
    img = img.resize(target_size)
    
    # Նկարը վերածում ենք numpy զանգվածի
    img_np = np.array(img)
    
    # Նկարը վերածում ենք PyTorch tensor-ի
    img_tensor = torch.tensor(img_np, dtype=torch.float32) / 255.0
    
    # Ձևափոխում ըստ մոդելի տեսակի
    if model_type == 'vgg16':
        # VGG16 համար: (1, 1, 224, 224) -> (1, 3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif model_type == 'cnn':
        # CNN համար: (1, 1, height, width)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    else:  # MLP համար
        # MLP համար: հարթեցնել
        img_tensor = img_tensor.view(1, -1)
    
    img_tensor = img_tensor.to(device)
    
    # Գուշակում
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        
        # ստանում ենք հավանականությունները
        probabilities = F.softmax(output, dim=1)
        
        # Ամենամեծ հավանականությունն ու դասը
        max_prob, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), max_prob.item()  # վերադարձնում ենք կանխատեսված դասը և հավանականությունը

# Keep the rest of the functions unchanged
def predict_proba(
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
        output = model(image)  # մոդելի կանխատեսում
        
        # ստանում ենք հավանականությունները
        probabilities = probabilities_function(output, dim=1)
        
        # Ամենամեծ հավանականությունն ու դասը
        max_prob, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), max_prob.item()  # վերադարձնում ենք կանխատեսված դասը և հավանականությունը

def visualize(
    image
):
    # Նկարի ձևափոխում՝ 28x28 չափսի մոխրագույն պատկեր
    image = image.view(28, 28).cpu().numpy()
    show_image(image, cmap='gray')
    return image

def load_image_from_url(url, label=None):
    """
    Ներբեռնում է նկարը URL-ից և վերածում այն մոխրագույն պատկեր
    """
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content))
    if label is None:
        # Եթե պիտակ չկա, ապա օգտագործում ենք URL-ի վերջին հատվածը որպես ֆայլի անուն
        filename = url.split("/")[-1] or "downloaded_image.jpg"
    else:
        filename = str(label)
    img.save(filename)  # պահում ենք որպես ժամանակավոր ֆայլ

    return img

def load_nst_model(model_url: str):
    """
    Load the TF‑Hub style transfer model.
    """
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")


def load_image_tf(path: str, max_dim: int = 512) -> tf.Tensor:
    """
    Load and preprocess image for hub model: float32 [1, h, w, 3], range [0,1].
    """
    img = Image.open(path).convert('RGB')
    long = max(img.size)
    scale = max_dim / long
    new_size = (round(img.size[0]*scale), round(img.size[1]*scale))
    img = img.resize(new_size)
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...].astype(np.float32)
    return tf.convert_to_tensor(img)


def stylize_image(model, content: tf.Tensor, style: tf.Tensor) -> tf.Tensor:
    """
    Run the hub model: returns stylized image tensor.
    """
    return model(tf.constant(content), tf.constant(style))[0]


def alpha_blend(content: tf.Tensor, stylized: tf.Tensor, alpha: float) -> tf.Tensor:
    """
    Blend content and stylized images by alpha.
    """
    return content * (1 - alpha) + stylized * alpha

