import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import numpy as np
import imageio.v3 as iio
import os
import glob
import torch.nn.functional as F
# 이미지 리사이징을 위해 scikit-image의 resize 함수를 사용합니다.
from skimage.transform import resize 

# ==============================================================================
# 0. EMoR 데이터 파싱 및 로드 (사용자 파일 포맷에 맞게 최종 수정)
# ==============================================================================

def parse_emor_data(file_path):
    """
    사용자의 'E =', 'f0 =', 'h(1) =', ..., 'h(25) =' 포맷에 맞게 데이터를 파싱합니다.
    """
    import os
    import numpy as np
    import torch
    import sys

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"오류: EMoR 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다.")

    with open(file_path, 'r') as f:
        # 줄 끝의 공백 제거 및 줄바꿈 문자 정규화
        lines = [line.strip().replace('\r', '') for line in f.readlines()] 

    # 1. 모든 태그의 시작 줄 인덱스를 찾습니다. (데이터는 다음 줄부터 시작)
    E_tag = 'E ='
    f0_tag = 'f0 ='
    
    # h(1) = 부터 h(25) = 까지 태그 리스트를 생성합니다.
    h_tags = [f'h({i})=' for i in range(1, 26)]
    all_tags = [E_tag, f0_tag] + h_tags
    
    tag_indices = {}
    
    for i, line in enumerate(lines):
        for tag in all_tags:
            if line.startswith(tag):
                tag_indices[tag] = i + 1
                break # 한 줄에 여러 태그가 있을 수 없으므로, 찾으면 다음 줄로 이동

    # 2. 필수 태그 27개(E, f0, h(1) ~ h(25))가 모두 있는지 확인합니다.
    if len(tag_indices) != 27:
        missing_tags = [tag for tag in all_tags if tag not in tag_indices]
        print(f"오류: 총 27개의 태그 중 {len(missing_tags)}개가 누락되었습니다: {missing_tags[:5]}...", file=sys.stderr)
        raise ValueError("EMoR 파일에서 필수 태그 27개 중 일부를 찾을 수 없습니다. 파일 포맷을 확인하십시오.")

    
    # 3. 라인 블록을 처리하여 넘파이 배열로 변환하는 헬퍼 함수
    def _process_lines(block_lines, count, tag_name=""):
        all_numbers = []
        for line in block_lines:
            if line: # 빈 줄이 아니면
                all_numbers.extend(line.split())

        # 문자열 리스트를 float 넘파이 배열로 변환
        data = np.float32(all_numbers[:count])
        
        if data.size < count:
            print(f"경고: {tag_name} 데이터 크기가 예상치({count})보다 작습니다. 실제 크기: {data.size}", file=sys.stderr)
            # 데이터가 부족하면 부족한 만큼 0으로 채워서 반환 (학습 진행을 위해)
            padded_data = np.zeros(count, dtype=np.float32)
            padded_data[:data.size] = data
            return padded_data
        
        return data

    
    # 4. E와 f0 데이터 추출 (1000개 샘플)
    E_start = tag_indices[E_tag]
    E_end = tag_indices[f0_tag] - 1
    E = _process_lines(lines[E_start:E_end], 1000, tag_name='E')

    f0_start = tag_indices[f0_tag]
    f0_end = tag_indices[h_tags[0]] - 1
    f0 = _process_lines(lines[f0_start:f0_end], 1000, tag_name='f0')


    # 5. H 행렬 추출 및 결합 (1000 x 25)
    H_components = []
    
    for k in range(25):
        current_tag = h_tags[k]
        
        # 다음 태그의 시작 인덱스를 찾습니다. (k=24일 때는 파일 끝 사용)
        if k < 24:
            next_tag = h_tags[k+1]
            H_end_idx = tag_indices[next_tag] - 1
        else:
            H_end_idx = len(lines)
            
        H_start_idx = tag_indices[current_tag]
        
        # h(k) 데이터 (1000개 샘플) 추출
        h_k = _process_lines(lines[H_start_idx:H_end_idx], 1000, tag_name=current_tag)
        H_components.append(h_k)
        
    # 25개의 (1000,) 벡터를 (1000, 25) 행렬로 결합 (25개의 열)
    H = np.stack(H_components, axis=1) 
    print(f"H 행렬 (PCA Basis) 파싱 완료. 크기: {H.shape}")

    # 6. Tensor 반환
    return torch.from_numpy(E).float(), torch.from_numpy(f0).float(), torch.from_numpy(H).float()

# ==============================================================================
# 1. 미분 가능한 TMO Layer (CRF Reconstruction)
# ==============================================================================

class DifferentiableTMO(nn.Module):
    def __init__(self, E_samples, f0_mean, H_basis):
        super().__init__()
        self.register_buffer('E_samples', E_samples) # (1000,)
        self.register_buffer('f0_mean', f0_mean)     # (1000,)
        self.register_buffer('H_basis', H_basis)     # (1000, 25)

    def forward(self, hdr_image, weights_w):
        # hdr_image: [B, 3, H, W] (원본 고해상도 HDR RGB)
        # weights_w: [B, 25] (PCA 가중치)
        
        B, C, H, W = hdr_image.shape
        
        # 1. CRF 곡선 생성 (CRF = f0 + H * w)
        # H_basis: [1000, 25]
        # weights_w.T: [25, B]
        # Matmul 결과: [1000, B]. Transpose하여 [B, 1000]
        curve_delta = torch.matmul(self.H_basis, weights_w.T).T 
        
        # [B, 1000] + [1000] (f0_mean) -> 브로드캐스트
        CRF_curve = self.f0_mean + curve_delta # [B, 1000]
        
        # 2. 픽셀 매핑 (보간)
        sdr_output = torch.zeros_like(hdr_image)
        
        # 각 배치 및 채널에 대해 CRF 보간 적용
        for i in range(B):
            for c in range(C):
                # (PLACEHOLDER: Differentiable Interpolation Logic)
                # **주의**: 이 부분은 np.interp를 사용하여 미분 불가능하며, 
                # 학습 시 경고가 발생합니다. 실제로는 PyTorch의 Differentiable 
                # Look-Up Table (LUT) 또는 Autograd Function으로 대체되어야 합니다.
                
                sdr_output[i, c, :, :] = self._interp_placeholder(
                    hdr_image[i, c, :, :],   # X_in: HDR 픽셀 값
                    self.E_samples,          # X_points: EMoR E samples
                    CRF_curve[i]             # Y_points: CRF curve
                )
        
        return torch.clamp(sdr_output, 0.0, 1.0)
    
    def _interp_placeholder(self, x_in, x_points, y_points):
        # np.interp는 미분 그래프를 끊으므로, detach() 후 numpy 연산 수행
        # 학습을 위한 개념 코드이므로 이대로 진행합니다.
        
        # TMO 적용 전 전역 스케일링 (Log-Avg Luminance 기반)이 필요하지만, 
        # 이는 TMO 알고리즘의 일부이므로, 여기서는 EMoR의 E_samples에 이미 
        # 정규화된 값이 입력된다고 가정하고 진행합니다. (Mean EMoR TMO 로직 생략)
        
        return torch.from_numpy(np.interp(x_in.detach().cpu().numpy(), 
                                          x_points.detach().cpu().numpy(), 
                                          y_points.detach().cpu().numpy()
                                         )).to(x_in.device).float()


# ==============================================================================
# 2. ResNet 기반 PCA Weight Predictor
# ==============================================================================

class ResNetEMoR(nn.Module):
    def __init__(self, E_samples, f0_mean, H_basis, output_weights=25):
        super().__init__()
        
        self.resnet = resnet18(weights=None)
        
        # 입력 채널 변경 (Luminance 1채널)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 최종 FC 레이어 변경 (512 -> 25 weights)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_weights)
        
        # Differentiable TMO Layer
        self.tmo_layer = DifferentiableTMO(E_samples, f0_mean, H_basis)

    def forward(self, hdr_luminance_input, hdr_rgb_full):
        # 1. PCA Weights 예측 (ResNet)
        weights_w = self.resnet(hdr_luminance_input) # [B, 25]
        
        # 2. CRF/TMO 재구성
        sdr_output = self.tmo_layer(hdr_rgb_full, weights_w) # [B, 3, H, W]
        
        return sdr_output, weights_w

# ==============================================================================
# 3. 데이터셋 및 전처리 (사용자 파일 구조 반영)
# ==============================================================================

class HDRLDRDataset(Dataset):
    def __init__(self, hdr_dir, ldr_dir, target_size=(256, 256), full_size=(1024, 1024)):
        self.hdr_dir = hdr_dir
        self.ldr_dir = ldr_dir
        self.target_size = target_size
        self.full_size = full_size
        
        # HDR 파일 목록에서 번호 추출 (e.g., '001', '002', ...)
        hdr_files = sorted(glob.glob(os.path.join(self.hdr_dir, 'HDR_*.hdr')))
        self.file_indices = [os.path.basename(f).split('_')[1].split('.')[0] for f in hdr_files]
        
        assert len(self.file_indices) > 0, f"오류: HDR 디렉토리에서 파일을 찾을 수 없습니다. 경로: {hdr_dir}"
        print(f"총 {len(self.file_indices)} 쌍의 이미지 인덱스 로드 준비 완료.")


    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_index = self.file_indices[idx]
        
        # 사용자 구조에 따른 파일 경로
        hdr_path = os.path.join(self.hdr_dir, f'HDR_{file_index}.hdr')
        ldr_path = os.path.join(self.ldr_dir, f'LDR_{file_index}.jpg') # LDR_exposure_0 폴더 내 LDR_XXX.jpg
        
        # 1. 원본 HDR 로드 (1024x1024, float)
        hdr_rgb_full = iio.imread(hdr_path).astype(np.float32)
        # LDR Ground Truth 로드 (0~1.0 float)
        ldr_gt_full = iio.imread(ldr_path).astype(np.float32) / 255.0
        
        # 2. ResNet 입력용 HDR 휘도 전처리
        # a) 휘도 추출 (Luminance)
        L_hdr_full = 0.2126 * hdr_rgb_full[..., 0] + 0.7152 * hdr_rgb_full[..., 1] + 0.0722 * hdr_rgb_full[..., 2]
        
        # b) 다운샘플링 (1024x1024 -> 256x256)
        # skimage.transform.resize 사용 (고품질 리사이징)
        L_hdr_downsampled = resize(L_hdr_full, self.target_size, 
                                   anti_aliasing=True, preserve_range=True).astype(np.float32)
        
        # c) 로그 변환 및 정규화 (log(L+eps))
        L_hdr_input = np.log(L_hdr_downsampled + 1e-5)
        # 데이터셋 전체 평균/표준편차로 정규화하는 것이 좋으나, 여기서는 이미지별 정규화 적용
        L_hdr_input = (L_hdr_input - L_hdr_input.mean()) / (L_hdr_input.std() + 1e-5)
        
        # Tensor 변환
        L_hdr_input = torch.from_numpy(L_hdr_input).unsqueeze(0) # [1, H', W']
        hdr_rgb_full_tensor = torch.from_numpy(hdr_rgb_full).permute(2, 0, 1) # [3, H, W]
        ldr_gt_full_tensor = torch.from_numpy(ldr_gt_full).permute(2, 0, 1) # [3, H, W]
        
        return L_hdr_input.float(), hdr_rgb_full_tensor.float(), ldr_gt_full_tensor.float()


# ==============================================================================
# 4. 학습 설정 및 실행
# ==============================================================================

# 🚨 사용자 지정 필수 1 🚨: LDR-HDR-pair_Dataset 폴더의 상위 경로를 지정해주세요.
DATASET_ROOT = os.path.expanduser('~/TM/') 

# 🚨 사용자 지정 필수 2 🚨: EMoR 데이터 파일의 절대 경로를 지정해주세요.
# 예시: EMOR_DATA_PATH = os.path.expanduser('/home/user/data/emorCurves.txt')
# 현재 코드가 있는 위치를 기준으로 상대 경로를 추정하는 코드입니다.
EMOR_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd(), '../dataset/emorCurves.txt'))

def train_model():
    # 데이터 경로 자동 구성
    HDR_DIR = os.path.join(DATASET_ROOT, 'LDR-HDR-pair_Dataset', 'HDR')
    LDR_DIR = os.path.join(DATASET_ROOT, 'LDR-HDR-pair_Dataset', 'LDR_exposure_0') # 사용자 구조 반영
    
    print(f"--- 데이터 경로 확인 ---")
    print(f"HDR 디렉토리: {HDR_DIR}")
    print(f"LDR 디렉토리: {LDR_DIR}")
    print(f"EMoR 파일: {EMOR_DATA_PATH}")
    print(f"------------------------")
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 1. EMoR 데이터 로드
    try:
        E_samples, f0_mean, H_basis = parse_emor_data(EMOR_DATA_PATH)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: EMoR 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {e}")
        return
    
    E_samples, f0_mean, H_basis = E_samples.to(device), f0_mean.to(device), H_basis.to(device)
    print(f"EMoR 데이터 로드 완료. H_basis.shape: {H_basis.shape}")

    # 2. 모델 초기화
    model = ResNetEMoR(E_samples, f0_mean, H_basis).to(device)
    
    # 3. 데이터 로더
    full_dataset = HDRLDRDataset(HDR_DIR, LDR_DIR)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"데이터셋 분할: 학습 {train_size}개, 검증 {val_size}개")
    
    # 4. 손실 함수 및 최적화
    criterion_recon = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    lambda_reg = 1e-5 # PCA Weight L2 정규화 가중치

    # 5. 학습 루프
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (L_hdr_input, hdr_rgb_full, ldr_gt_full) in enumerate(train_loader):
            L_hdr_input, hdr_rgb_full, ldr_gt_full = L_hdr_input.to(device), hdr_rgb_full.to(device), ldr_gt_full.to(device)
            
            optimizer.zero_grad()
            
            sdr_pred, weights_w = model(L_hdr_input, hdr_rgb_full)
            
            # L1 재구성 손실
            loss_recon = criterion_recon(sdr_pred, ldr_gt_full)
            
            # L2 Weight 정규화 손실
            loss_reg = torch.mean(weights_w.pow(2))
            
            loss = loss_recon + lambda_reg * loss_reg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * L_hdr_input.size(0)
            
        epoch_loss = running_loss / train_size
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f} (Recon: {loss_recon.item():.6f}, Reg: {loss_reg.item():.6f})")
        
        # 6. 검증
        evaluate_model(model, val_loader, device, val_size)


# ==============================================================================
# 5. 평가 함수 (TMQI Placeholder)
# ==============================================================================

def evaluate_model(model, val_loader, device, val_size):
    model.eval()
    val_loss = 0.0
    # TMQI PLACEHOLDER: tmqi_scores는 현재 계산되지 않습니다.
    
    with torch.no_grad():
        for L_hdr_input, hdr_rgb_full, ldr_gt_full in val_loader:
            L_hdr_input, hdr_rgb_full, ldr_gt_full = L_hdr_input.to(device), hdr_rgb_full.to(device), ldr_gt_full.to(device)
            
            sdr_pred, weights_w = model(L_hdr_input, hdr_rgb_full)
            
            loss_recon = nn.L1Loss()(sdr_pred, ldr_gt_full)
            val_loss += loss_recon.item() * L_hdr_input.size(0)
            
            # ------------------------------------------------------------------
            # TMQI 계산 로직이 들어갈 자리 (현재는 L1 Loss만 측정)
            # tmqi_score = calculate_tmqi(hdr_rgb_full, sdr_pred) 
            # ------------------------------------------------------------------

    avg_val_loss = val_loss / val_size
    print(f"  Validation L1 Loss: {avg_val_loss:.6f} (TMQI 미측정)")
    
    model.train()

if __name__ == '__main__':
    train_model()