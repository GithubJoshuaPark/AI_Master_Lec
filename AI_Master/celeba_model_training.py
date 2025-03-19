import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm


# CelebADataset 클래스 정의 - 속성 예측을 위한 버전
class CelebAAttributeDataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None, target_attributes=None):
        self.img_dir = img_dir
        self.transform = transform
        
        if not os.path.isfile(attr_file):
            raise FileNotFoundError(f"Attribute file '{attr_file}' not found.")
        
        # 속성 파일 읽기
        with open(attr_file, 'r') as f:
            num_images = int(f.readline().strip())
            attr_names = f.readline().strip().split()
        
        # 속성 데이터프레임 생성
        self.attr_df = pd.read_csv(attr_file, sep=r'\s+', skiprows=2, names=['Image'] + attr_names)
        
        # 타겟 속성 설정 (기본값: 모든 속성)
        self.target_attributes = target_attributes if target_attributes else attr_names
        
        # 속성값을 0과 1로 변환 (-1 -> 0, 1 -> 1)
        for attr in self.target_attributes:
            if attr in self.attr_df.columns:
                self.attr_df[attr] = (self.attr_df[attr] + 1) // 2
        
    def __len__(self):
        return len(self.attr_df)
    
    def __getitem__(self, idx):
        img_name = self.attr_df.iloc[idx]['Image']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{img_path}' not found.")
        
        if self.transform:
            image = self.transform(image)
        
        # 타겟 속성 추출
        attributes = torch.tensor(self.attr_df.iloc[idx][self.target_attributes].values, dtype=torch.float32)
        
        return image, attributes, img_name

# CNN 모델 정의
class AttributeClassifier(nn.Module):
    def __init__(self, num_attributes):
        super(AttributeClassifier, self).__init__()
        
        # ResNet18을 기반 모델로 사용
        self.base_model = models.resnet18(pretrained=True)
        
        # 마지막 FC 레이어를 속성 수에 맞게 변경
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_attributes),
            nn.Sigmoid()  # 각 속성은 이진 분류이므로 시그모이드 사용
        )
    
    def forward(self, x):
        return self.base_model(x)

# 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        running_loss = 0.0
        
        # 프로그레스 바 설정
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, attributes, _ in train_bar:
            images = images.to(device)
            attributes = attributes.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, attributes)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 검증 모드
        model.eval()
        running_val_loss = 0.0
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, attributes, _ in val_bar:
                images = images.to(device)
                attributes = attributes.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, attributes)
                
                running_val_loss += loss.item() * images.size(0)
                val_bar.set_postfix(loss=loss.item())
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        # 최고 모델 저장
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_celeba_model.pth')
            print(f'Model saved with validation loss: {best_val_loss:.4f}')
    
    return model, train_losses, val_losses

# 모델 평가 함수
def evaluate_model(model, test_loader, criterion, device='cuda'):
    model = model.to(device)
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, attributes, _ in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            attributes = attributes.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, attributes)
            
            running_loss += loss.item() * images.size(0)
            
            # 예측값 저장 (0.5를 기준으로 이진 분류)
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(attributes.cpu())
    
    # 전체 손실 계산
    test_loss = running_loss / len(test_loader.dataset)
    
    # 예측 결과 합치기
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 정확도 계산
    accuracy = (all_preds == all_labels).float().mean()
    
    # 속성별 정확도 계산
    attr_accuracy = (all_preds == all_labels).float().mean(dim=0)
    
    return test_loss, accuracy, attr_accuracy

# 예측 결과 시각화 함수
def visualize_predictions(model, test_loader, target_attributes, device='cuda', num_samples=5):
    model = model.to(device)
    model.eval()
    
    # 샘플 이미지와 레이블 가져오기
    images_list = []
    labels_list = []
    names_list = []
    
    with torch.no_grad():
        for images, attributes, names in test_loader:
            images_list.append(images)
            labels_list.append(attributes)
            names_list.extend(names)
            
            if len(images_list) * images.size(0) >= num_samples:
                break
    
    # 샘플 데이터 합치기
    images = torch.cat(images_list, dim=0)[:num_samples]
    labels = torch.cat(labels_list, dim=0)[:num_samples]
    names = names_list[:num_samples]
    
    # 예측 수행
    images_device = images.to(device)
    predictions = model(images_device)
    binary_preds = (predictions > 0.5).float().cpu()
    
    # 결과 시각화
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # 이미지 표시
        ax_img = axes[i] if num_samples > 1 else axes
        ax_img.imshow(images[i].permute(1, 2, 0))
        ax_img.set_title(f'Image: {names[i]}')
        ax_img.axis('off')
        
        # 예측 결과 텍스트로 표시
        result_text = ""
        for j, attr in enumerate(target_attributes):
            true_val = "Yes" if labels[i, j] == 1 else "No"
            pred_val = "Yes" if binary_preds[i, j] == 1 else "No"
            result_text += f"{attr}: True={true_val}, Pred={pred_val}\n"
        
        ax_img.text(1.05, 0.5, result_text, transform=ax_img.transAxes, fontsize=12, 
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

# 학습 곡선 시각화 함수
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# 메인 실행 코드
def main():
    # 데이터 경로 설정
    base_dir = './data/celeba'
    img_dir = os.path.join(base_dir, 'img_align_celeba')
    attr_file = os.path.join(base_dir, 'list_attr_celeba.csv')
    
    # 디렉토리와 파일 존재 확인
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory '{img_dir}' not found.")
    if not os.path.isfile(attr_file):
        raise FileNotFoundError(f"Attribute file '{attr_file}' not found.")
    
    # 타겟 속성 설정 (관심 있는 속성만 선택)
    target_attributes = ['Male', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 
                         'Mustache', 'Wearing_Hat', 'Smiling']
    
    # 변환 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])
    
    # 데이터셋 생성
    full_dataset = CelebAAttributeDataset(
        img_dir=img_dir, 
        attr_file=attr_file, 
        transform=transform,
        target_attributes=target_attributes
    )
    
    # 데이터셋 분할 (학습:검증:테스트 = 7:1:2)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드 설정
    )
    
    # 데이터로더 생성
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = AttributeClassifier(num_attributes=len(target_attributes))
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 모델 학습
    num_epochs = 5  # 실제 학습에서는 더 많은 에폭 사용 권장
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    # 학습 곡선 시각화
    plot_training_curves(train_losses, val_losses)
    
    # 최고 모델 로드
    model.load_state_dict(torch.load('best_celeba_model.pth'))
    
    # 테스트 세트에서 모델 평가
    test_loss, accuracy, attr_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Overall Accuracy: {accuracy:.4f}")
    
    # 속성별 정확도 출력
    for i, attr in enumerate(target_attributes):
        print(f"Accuracy for {attr}: {attr_accuracy[i]:.4f}")
    
    # 예측 결과 시각화
    visualize_predictions(model, test_loader, target_attributes, device)

if __name__ == "__main__":
    main()
