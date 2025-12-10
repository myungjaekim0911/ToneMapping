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
import matplotlib.pyplot as plt
import sys

# ==============================================================================
# 0. EMoR 데이터 파싱 및 로드 (이전 버전과 동일)
# ==============================================================================

def parse_emor_data(file_path):
    """
    사용자의 'E =', 'f0 =', 'h(1) =', ..., 'h(25) =' 포맷에 맞게 데이터를 파싱합니다.
    """
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
                break

    # 2. 필수 태그 27개(E, f0, h(1) ~ h(25))가 모두 있는지 확인합니다.
    if len(tag_indices) != 27:
        missing_tags = [tag for tag in all_tags if tag not in tag_indices]
        print(f"오류: 총 27개의 태그 중 {len(missing_tags)}개가 누락되었습니다: {missing_tags[:5]}...", file=sys.stderr)
        raise ValueError("EMoR 파일에서 필수 태그 27개 중 일부를 찾을 수 없습니다. 파일 포맷을 확인하십시오.")

    
    # 3. 라인 블록을 처리하여 넘파이 배열로 변환하는 헬퍼 함수
    def _process_lines(block_lines, count, tag_name=""):
        all_numbers = []
        for line in block_lines:
            if line:
                all_numbers.extend(line.split())

        data = np.float32(all_numbers[:count])
        
        if data.size < count:
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
        
        if k < 24:
            next_tag = h_tags[k+1]
            H_end_idx = tag_indices[next_tag] - 1
        else:
            H_end_idx = len(lines)
            
        H_start_idx = tag_indices[current_tag]
        
        h_k = _process_lines(lines[H_start_idx:H_end_idx], 1000, tag_name=current_tag)
        H_components.append(h_k)
        
    H = np.stack(H_components, axis=1) 
    print(f"H 행렬 (PCA Basis) 파싱 완료. 크기: {H.shape}")

    return torch.from_numpy(E).float(), torch.from_numpy(f0).float(), torch.from_numpy(H).float()

# ==============================================================================
# 1. 미분 가능한 TMO Layer (CRF Reconstruction) - 이전 버전과 동일
# ==============================================================================

class DifferentiableTMO(nn.Module):
    def __init__(self, E_samples, f0_mean, H_basis):
        super().__init__()
        self.register_buffer('E_samples', E_samples) # (1000,)
        self.register_buffer('f0_mean', f0_mean)     # (1000,)
        self.register_buffer('H_basis', H_basis)     # (1000, 25)

    def forward(self, hdr_image, weights_w):
        
        B, C, H, W = hdr_image.shape
        
        # 1. CRF 곡선 생성 (CRF = f0 + H * w)
        curve_delta = torch.matmul(self.H_basis, weights_w.T).T 
        CRF_curve = self.f0_mean + curve_delta # [B, 1000]
        
        # 2. 픽셀 매핑 (보간)
        sdr_output = torch.zeros_like(hdr_image)
        
        for i in range(B):
            for c in range(C):
                sdr_output[i, c, :, :] = self._interp_placeholder(
                    hdr_image[i, c, :, :],   # X_in: HDR 픽셀 값
                    self.E_samples,          # X_points: EMoR E samples
                    CRF_curve[i]             # Y_points: CRF curve
                )
        
        return torch.clamp(sdr_output, 0.0, 1.0)
    
    def _interp_placeholder(self, x_in, x_points, y_points):
        # 경고를 발생시키는 미분 불가능한 np.interp 사용
        return torch.from_numpy(np.interp(x_in.detach().cpu().numpy(), 
                                          x_points.detach().cpu().numpy(), 
                                          y_points.detach().cpu().numpy()
                                         )).to(x_in.device).float()


# ==============================================================================
# 2. ResNet 기반 PCA Weight Predictor - 이전 버전과 동일
# ==============================================================================

class ResNetEMoR(nn.Module):
    def __init__(self, E_samples, f0_mean, H_basis, output_weights=25):
        super().__init__()
        
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_weights)
        
        self.tmo_layer = DifferentiableTMO(E_samples, f0_mean, H_basis)

    def forward(self, hdr_luminance_input, hdr_rgb_full):
        weights_w = self.resnet(hdr_luminance_input) # [B, 25]
        sdr_output = self.tmo_layer(hdr_rgb_full, weights_w) # [B, 3, H, W]
        return sdr_output, weights_w

# ==============================================================================
# 3. 데이터셋 및 전처리 - 이전 버전과 동일
# ==============================================================================

class HDRLDRDataset(Dataset):
    def __init__(self, hdr_dir, ldr_dir, target_size=(256, 256), full_size=(1024, 1024)):
        self.hdr_dir = hdr_dir
        self.ldr_dir = ldr_dir
        self.target_size = target_size
        self.full_size = full_size
        
        hdr_files = sorted(glob.glob(os.path.join(self.hdr_dir, 'HDR_*.hdr')))
        self.file_indices = [os.path.basename(f).split('_')[1].split('.')[0] for f in hdr_files]
        
        assert len(self.file_indices) > 0, f"오류: HDR 디렉토리에서 파일을 찾을 수 없습니다. 경로: {hdr_dir}"
        print(f"총 {len(self.file_indices)} 쌍의 이미지 인덱스 로드 준비 완료.")


    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_index = self.file_indices[idx]
        
        hdr_path = os.path.join(self.hdr_dir, f'HDR_{file_index}.hdr')
        ldr_path = os.path.join(self.ldr_dir, f'LDR_{file_index}.jpg')
        
        hdr_rgb_full = iio.imread(hdr_path).astype(np.float32)
        ldr_gt_full = iio.imread(ldr_path).astype(np.float32) / 255.0
        
        # 휘도 추출 및 다운샘플링
        L_hdr_full = 0.2126 * hdr_rgb_full[..., 0] + 0.7152 * hdr_rgb_full[..., 1] + 0.0722 * hdr_rgb_full[..., 2]
        L_hdr_downsampled = resize(L_hdr_full, self.target_size, 
                                   anti_aliasing=True, preserve_range=True).astype(np.float32)
        
        # 로그 변환 및 정규화
        L_hdr_input = np.log(L_hdr_downsampled + 1e-5)
        L_hdr_input = (L_hdr_input - L_hdr_input.mean()) / (L_hdr_input.std() + 1e-5)
        
        L_hdr_input = torch.from_numpy(L_hdr_input).unsqueeze(0)
        hdr_rgb_full_tensor = torch.from_numpy(hdr_rgb_full).permute(2, 0, 1)
        ldr_gt_full_tensor = torch.from_numpy(ldr_gt_full).permute(2, 0, 1)
        
        return L_hdr_input.float(), hdr_rgb_full_tensor.float(), ldr_gt_full_tensor.float()


# ==============================================================================
# 5. 평가 지표 함수 (TMQI Proxy)
# ==============================================================================

def calculate_tmqi_proxy(sdr_pred, ldr_gt_full):
    """
    TMQI의 구조적 품질(S)을 근사하기 위해 로그 휘도 도메인에서 MSE를 사용하여
    TMQI 프록시 점수를 계산합니다. (점수는 0~1, 1이 최적)
    """
    
    def get_luminance(img_tensor): # [B, 3, H, W]
        # Rec. 709 휘도 공식: 0.2126R + 0.7152G + 0.0722B
        R, G, B = img_tensor.unbind(1)
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        return L.unsqueeze(1) # [B, 1, H, W]

    L_pred = get_luminance(sdr_pred)
    L_gt = get_luminance(ldr_gt_full)
    
    eps = 1e-5
    
    # 로그 변환
    Log_L_pred = torch.log(L_pred + eps)
    Log_L_gt = torch.log(L_gt + eps)
    
    # 구조적 손실(MSE)
    loss_S = F.mse_loss(Log_L_pred, Log_L_gt)
    
    # TMQI 스코어 변환: Score = exp(-k * Loss) (점수를 0~1 범위로 매핑)
    # k=10을 사용하여 작은 손실을 점수로 변환
    S_score = torch.exp(-10 * loss_S) 
    
    # Naturalness (N) component는 복잡하므로 생략하고 S_score만 반환
    return S_score.mean().item()


def evaluate_model(model, val_loader, device, val_size):
    model.eval()
    val_loss = 0.0
    val_tmqi_total = 0.0
    
    with torch.no_grad():
        for L_hdr_input, hdr_rgb_full, ldr_gt_full in val_loader:
            L_hdr_input, hdr_rgb_full, ldr_gt_full = L_hdr_input.to(device), hdr_rgb_full.to(device), ldr_gt_full.to(device)
            
            sdr_pred, weights_w = model(L_hdr_input, hdr_rgb_full)
            
            loss_recon = nn.L1Loss()(sdr_pred, ldr_gt_full)
            val_loss += loss_recon.item() * L_hdr_input.size(0)
            
            # TMQI 계산
            val_tmqi_total += calculate_tmqi_proxy(sdr_pred, ldr_gt_full) * L_hdr_input.size(0)

    avg_val_loss = val_loss / val_size
    avg_val_tmqi = val_tmqi_total / val_size
    print(f"  Validation L1 Loss: {avg_val_loss:.6f}, TMQI Score: {avg_val_tmqi:.6f}")
    
    model.train()
    return avg_val_loss, avg_val_tmqi

# ==============================================================================
# 6. 학습된 EMoR Curve 및 지표 추이 시각화 함수 (수정됨)
# ==============================================================================

def plot_results(model, val_loader, E_samples, f0_mean, H_basis, device, loss_history, tmqi_history):
    """
    1. 학습된 EMoR Curve를 시각화합니다.
    2. 학습/검증 지표(L1 Loss, TMQI) 추이를 시각화합니다.
    """
    # ------------------ 1. EMoR Curve 시각화 ------------------
    model.eval()
    L_hdr_input, hdr_rgb_full, _ = next(iter(val_loader))
    L_hdr_input, hdr_rgb_full = L_hdr_input.to(device), hdr_rgb_full.to(device)
    
    with torch.no_grad():
        _, weights_w = model(L_hdr_input[0].unsqueeze(0), hdr_rgb_full[0].unsqueeze(0)) 

    w_vector = weights_w.squeeze(0).cpu().numpy()
    E_numpy = E_samples.cpu().numpy()
    f0_numpy = f0_mean.cpu().numpy()
    H_numpy = H_basis.cpu().numpy()

    curve_residual = H_numpy.dot(w_vector)
    final_crf_curve = f0_numpy + curve_residual

    plt.figure(figsize=(18, 6))
    
    # 1-1. EMoR Curve
    plt.subplot(1, 2, 1)
    plt.plot(E_numpy, final_crf_curve, label='Learned Tone Mapping Curve', color='red', linewidth=3)
    plt.plot(E_numpy, f0_numpy, '--', label='EMoR Mean Curve ($\mathbf{f}_0$)', color='gray', alpha=0.7)
    plt.xlabel('Scene Linear Radiance')
    plt.ylabel('LDR Pixel Value')
    plt.title('Learned EMoR Tone Mapping Curve (from a Single Validation Image)')
    plt.grid(True)
    plt.legend()
    
    # ------------------ 2. 지표 추이 시각화 ------------------
    epochs = range(1, len(loss_history) + 1)
    
    # 1-2. Loss 및 TMQI 추이
    plt.subplot(1, 2, 2)
    
    # Loss Plot
    line1, = plt.plot(epochs, loss_history, 
                      label='Validation L1 Loss (Minimize)', # <--- 라벨 수정
                      color='blue', marker='o', linestyle='-')
    
    # TMQI Plot (TMQI는 값이 클수록 좋으므로 오른쪽 Y축 사용)
    ax2 = plt.gca().twinx()
    line2, = ax2.plot(epochs, tmqi_history, 
                       label='Validation TMQI Score (Maximize)', # <--- 라벨 수정
                       color='green', marker='x', linestyle='--')
    
    plt.xlabel('Epoch')
    
    # Y축 라벨은 그대로 유지 (ax2의 중복 라벨 설정은 제거)
    plt.ylabel('Validation L1 Loss (Minimize)', color='blue')
    ax2.set_ylabel('Validation TMQI Score (Maximize)', color='green') # ax2에 대한 라벨만 설정

    plt.title('Validation Metrics Over Epochs')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # === 통합된 범례 생성 및 오른쪽 위에 위치시키기 ===
    # 두 플롯 객체(line1, line2)와 라벨을 통합하여 하나의 범례를 생성
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    
    # 오른쪽 상단에 통합 범례 표시
    plt.legend(lines, labels, loc='upper right') 
    
    # 기존의 분리된 범례 호출은 제거합니다.
    # plt.legend(loc='upper left') 
    # ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    model.train()

# ==============================================================================
# 4. 학습 설정 및 실행 (수정됨)
# ==============================================================================

# 🚨 사용자 지정 필수 🚨: LDR-HDR-pair_Dataset 폴더의 상위 경로와 EMoR 파일 경로를 확인해주세요.
DATASET_ROOT = os.path.expanduser('~/TM/') 
EMOR_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd(), '../dataset/emorCurves.txt'))

def train_model():
    HDR_DIR = os.path.join(DATASET_ROOT, 'LDR-HDR-pair_Dataset', 'HDR')
    LDR_DIR = os.path.join(DATASET_ROOT, 'LDR-HDR-pair_Dataset', 'LDR_exposure_0')
    
    print(f"--- 데이터 경로 확인 ---")
    print(f"HDR 디렉토리: {HDR_DIR}")
    print(f"LDR 디렉토리: {LDR_DIR}")
    print(f"EMoR 파일: {EMOR_DATA_PATH}")
    print(f"------------------------")
    
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
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    lambda_reg = 1e-5
    
    # 5. 지표 저장 리스트
    val_loss_history = []
    val_tmqi_history = []

    # 6. 학습 루프
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (L_hdr_input, hdr_rgb_full, ldr_gt_full) in enumerate(train_loader):
            L_hdr_input, hdr_rgb_full, ldr_gt_full = L_hdr_input.to(device), hdr_rgb_full.to(device), ldr_gt_full.to(device)
            
            optimizer.zero_grad()
            
            sdr_pred, weights_w = model(L_hdr_input, hdr_rgb_full)
            
            loss_recon = criterion_recon(sdr_pred, ldr_gt_full)
            loss_reg = torch.mean(weights_w.pow(2))
            
            loss = loss_recon + lambda_reg * loss_reg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * L_hdr_input.size(0)
            
        epoch_loss = running_loss / train_size
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f} (Recon: {loss_recon.item():.6f}, Reg: {loss_reg.item():.6f})")
        
        # 7. 검증 및 지표 저장
        avg_val_loss, avg_val_tmqi = evaluate_model(model, val_loader, device, val_size)
        val_loss_history.append(avg_val_loss)
        val_tmqi_history.append(avg_val_tmqi)

    # 8. 학습 완료 후, EMoR Curve 및 지표 추이 시각화
    print("\n--- 학습 완료. EMoR Curve 및 지표 추이 시각화를 시작합니다 ---")
    plot_results(model, val_loader, E_samples, f0_mean, H_basis, device, val_loss_history, val_tmqi_history)


if __name__ == '__main__':
    train_model()