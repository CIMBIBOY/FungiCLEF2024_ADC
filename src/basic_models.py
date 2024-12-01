from datasets.fungidataset import build_dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from torchvision import models 
import torch
import torch.nn as nn
import numpy as np
import sklearn
from tqdm import tqdm
IMAGEDIR = "/Users/koksziszdave/Downloads/fungi_images"
LABELDIR = "/Users/koksziszdave/Downloads/fungi_train_metadata.csv"

args = {
    "image_dir": IMAGEDIR,
    "labels_path": LABELDIR,
    "pre_load": False,
    "batch_size": 32
}

train_loader, valid_loader = build_dataset(args)

num_semcls = 100
def extract_features(model, dataloader, device):
    """
    Extract features using a pre-trained CNN.
    """
    model.eval()
    features, labels, poisonous = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if(batch["image"].shape[0]!=32):
                continue
            images, lbls, poi= batch["image"].to(device), batch["target_sem_cls"].to(device) , batch["target_poisonous"].to(device)
            
            images=images.permute(0,3,1,2)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(lbls.cpu().numpy())
            poisonous.append(poi.cpu().numpy())
            

    features = np.vstack(features)
    labels = np.vstack(labels)
    poisonous = np.vstack(poisonous)

    return features, labels, poisonous

def train_model(train_loader, val_loader):
    """
    Train a model using deep features and AdaBoostClassifier.
    """
    # Use a pre-trained ResNet for feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()  # Remove the classification head
    resnet = resnet.to(device)

    # Extract features
    print("Extracting features from the training set...")
    X_train, y_train ,y_train_pois = extract_features(resnet, train_loader, device)
    print("Extracting features from the validation set...")
    X_val, y_val ,y_val_pois = extract_features(resnet, val_loader, device)
    y_val_pois=y_val_pois.astype(int)
    
    y_train_pois=y_train_pois.astype(int)

    print(X_train.shape, y_train.shape, y_train_pois.shape)
    print(X_val.shape, y_val.shape, y_val_pois.shape)
    #Save to numpy file
    np.save("X_train.npy",X_train)
    np.save("y_train.npy",y_train)
    np.save("y_train_pois.npy",y_train_pois)
    np.save("X_val.npy",X_val)
    np.save("y_val.npy",y_val)
    np.save("y_val_pois.npy",y_val_pois)

    print("Features extracted successfully!")
    y_train_poisonous = y_train_pois.squeeze()  # Now shape: (32,)
    y_val_poisonous = y_val_pois.squeeze()      # Now shape: (32,)

    # Convert one-hot classes to integers
    y_train_classes = np.argmax(y_train, axis=1)  # Now shape: (32,)
    y_val_classes = np.argmax(y_val, axis=1)      # Now shape: (32,)

    # Train models
    print("Training the classification model...")
    class_model = OneVsRestClassifier(AdaBoostClassifier(n_estimators=50))
    class_model.fit(X_train, y_train_classes)

    print("Training the poisonousness model...")
    poison_model = AdaBoostClassifier(n_estimators=50)
    poison_model.fit(X_train, y_train_poisonous)

    # Evaluate models
    class_preds = class_model.predict(X_val)
    poison_preds = poison_model.predict(X_val)

    print("Classification Report for Classes:")
    print(classification_report(y_val_classes, class_preds))

    print("Classification Report for Poisonousness:")
    print(classification_report(y_val_poisonous, poison_preds))

train_model(train_loader, valid_loader)