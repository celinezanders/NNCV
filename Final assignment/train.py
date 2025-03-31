"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)

from unet import Model

#toevoegen voor mixed precision training
scaler = torch.amp.GradScaler("cuda")


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        #Normalize((0.5,), (0.5,)),
        # above it treated the RGB in grayscale, below as RGB
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    #criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

# I am adding the dice loss to the loss function
    def dice_loss(pred, target, smooth=1e-6):
        pred = torch.softmax(pred, dim=1)  # softmax over classes
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])  # [batch, H, W, num_classes]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # Maak het [batch, num_classes, H, W]

        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))  # Som over hoogte/breedte
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))  # Som over hoogte/breedte
        #intersection = torch.sum(pred * target)
        #union = torch.sum(pred) + torch.sum(target)

        #return 1 - (2. * intersection + smooth) / (union + smooth)

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    criterion = lambda output, target: nn.CrossEntropyLoss(ignore_index=255)(output, target) + 0.3*dice_loss(output, target)
#added 0.3* to dice_loss to reduce the weight on dice which can make the loss too sensitive
    
    # Define the optimizer
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#scheduler toegevoed voor betere learningrate
   # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension
            labels[labels == 255] = 0

            optimizer.zero_grad()
# hier voeg ik Mixed Precision Training toe            
            #outputs = model(images)
            #loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()
            with torch.amp.autocast("cuda"): # Zet automatische mixed precision aan
                outputs = model(images)  
                loss = criterion(outputs, labels) 
                

            scaler.scale(loss).backward()  # Schaal de gradiënten om stabiliteit te garanderen
            scaler.step(optimizer)  # Update de optimizer
            scaler.update()  # Pas de scaler aan voor de volgende iteratie

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension
                labels[labels == 255] = 0

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
           # valid_loss = sum(losses) / len(losses)
           # wandb.log({
           #     "valid_loss": valid_loss
           # }, step=(epoch + 1) * len(train_dataloader) - 1)

            valid_loss = sum(losses) / len(losses)
            #early stopping to prevent overfitting
            # Early stopping criteria
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0  # Reset patience counter als er verbetering is
            else:
                patience_counter += 1  # Verhoog patience als valid_loss niet verbetert

            # Stop de training als valid_loss niet meer verbetert na X epochs
            if patience_counter >= 10:  # Stop na 10 epochs zonder verbetering
                print("Early stopping triggered! Training stopped.")
                break

# scheduler toegevoegd voor betere learning rate 
            scheduler.step(valid_loss)

            #hier dice formule toevoegen
            predictions = outputs.softmax(1).argmax(1) # logits naar klassenvoorspellingen
            intersection = torch.sum((predictions == labels) * (labels != 255)) #negeer index 255
            union = torch.sum((predictions!= 255)) + torch.sum((labels != 255))
            dice_score = (2.0 * intersection) / union if union >0 else 1.0 # zodat er geen deling door nul gebeurt

            wandb.log({
                "valid_loss": valid_loss,
                "dice_score": dice_score.item()
            }, step=(epoch + 1) * len(train_dataloader) - 1)


            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)