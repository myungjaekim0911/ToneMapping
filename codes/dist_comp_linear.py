import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os

# --- 파일 경로 설정 (사용자 요청 반영 및 안정화) ---
RAW_HDR_FILE_PATH = '~/TM/LDR-HDR-pair_Dataset/HDR/HDR_001.hdr' 
RAW_EMOR_FILE_PATH = '../dataset/emorCurves.txt'
RAW_OUTPUT_DIR = '~/TM/temp_results'
OUTPUT_JPEG_FILENAME = 'output_mean_emor_001.jpg'

# 경로 확장 및 구성
HDR_FILE_PATH = os.path.expanduser(RAW_HDR_FILE_PATH)
OUTPUT_DIR = os.path.expanduser(RAW_OUTPUT_DIR)
SDR_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_JPEG_FILENAME)
PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'linear_normalized_luminance_plot.png')
# EMoR 파일 경로는 이 분석 스크립트에서 직접 사용되지 않으나, 경로 안정화는 유지합니다.
EMOR_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in locals() else os.getcwd(), RAW_EMOR_FILE_PATH))


# --- 1. 휘도(Luminance) 계산 함수 ---
def compute_luminance(image_data):
    """
    RGB 데이터로부터 ITU-R BT.709 표준을 기반으로 휘도를 계산합니다.
    """
    # 0.2126R + 0.7152G + 0.0722B
    return 0.2126 * image_data[..., 0] + 0.7152 * image_data[..., 1] + 0.0722 * image_data[..., 2]

# --- 2. 분포 분석 및 플롯 함수 ---
def plot_linear_normalized_distribution(hdr_data, sdr_data):
    """
    HDR과 SDR 이미지의 휘도 분포를 각각 선형 정규화된 X축에 플롯하고,
    클리핑 비율을 표시합니다.
    """
    
    # 1. 휘도 추출 및 평탄화
    L_hdr = compute_luminance(hdr_data).flatten()
    L_sdr = compute_luminance(sdr_data).flatten()
    
    # --- 2. HDR 데이터 분석 및 정규화 ---
    
    # HDR의 휘도 범위
    hdr_min = L_hdr.min()
    hdr_max = L_hdr.max()
    total_pixels = L_hdr.size
    
    # 클리핑 비율 계산 (최대/최소값에 있는 픽셀)
    # float 비교를 위해 작은 허용 오차(epsilon)를 사용
    epsilon = 1e-4
    
    # 최대 휘도에 클리핑된 픽셀 비율
    hdr_white_clipped_count = np.sum(L_hdr >= hdr_max - epsilon)
    hdr_white_clipped_percent = (hdr_white_clipped_count / total_pixels) * 100
    
    # 최소 휘도에 클리핑된 픽셀 비율
    hdr_black_clipped_count = np.sum(L_hdr <= hdr_min + epsilon)
    hdr_black_clipped_percent = (hdr_black_clipped_count / total_pixels) * 100
    
    # 선형 정규화된 HDR 데이터
    if (hdr_max - hdr_min) < epsilon:
         # 다이내믹 레인지가 거의 없는 경우 (매우 드뭄)
         hdr_normalized = np.zeros_like(L_hdr)
    else:
         hdr_normalized = (L_hdr - hdr_min) / (hdr_max - hdr_min)
    
    # --- 3. SDR 데이터 분석 및 정규화 ---
    
    # SDR은 0.0 ~ 1.0 범위
    sdr_min = L_sdr.min() # 거의 0.0
    sdr_max = L_sdr.max() # 거의 1.0
    
    # 클리핑 비율 계산 (0과 1.0에 있는 픽셀)
    # TMO 출력은 0.0 ~ 1.0 범위로 강제되므로, 이 값들이 손실 픽셀임.
    sdr_white_clipped_count = np.sum(L_sdr >= 1.0 - epsilon)
    sdr_white_clipped_percent = (sdr_white_clipped_count / total_pixels) * 100
    
    sdr_black_clipped_count = np.sum(L_sdr <= 0.0 + epsilon)
    sdr_black_clipped_percent = (sdr_black_clipped_count / total_pixels) * 100
    
    # SDR은 이미 0~1 사이이므로, X축은 그 자체로 사용됩니다.
    sdr_normalized = L_sdr
    
    # --- 4. 히스토그램 계산 ---
    
    # HDR 히스토그램 (선형 정규화된 X축 사용)
    hdr_hist, bins = np.histogram(hdr_normalized, bins=256, range=(0.0, 1.0), density=True)
    # SDR 히스토그램 (선형 스케일 사용)
    sdr_hist, _ = np.histogram(sdr_normalized, bins=256, range=(0.0, 1.0), density=True)
    
    # X축 값 (빈 중앙값)
    centers = (bins[:-1] + bins[1:]) / 2
    
    # --- 5. 플롯 생성 ---
    
    plt.figure(figsize=(12, 7))
    
    # 1. HDR 휘도 분포 플롯
    plt.plot(centers, hdr_hist, 
             label=f'HDR (Input) Distribution', 
             color='blue', linewidth=2, alpha=0.7)
    
    # 2. SDR 휘도 분포 플롯
    plt.plot(centers, sdr_hist, 
             label=f'SDR (TMO Output) Distribution', 
             color='red', linestyle='--', linewidth=2, alpha=0.9)
    
    # 3. 텍스트 정보 표시
    text_y_offset = plt.ylim()[1] * 0.95
    
    '''
    # HDR 클리핑 정보
    plt.text(0.02, text_y_offset, 
             f'HDR Max Clipped: {hdr_white_clipped_percent:.2f}%\nHDR Min Clipped: {hdr_black_clipped_percent:.2f}%', 
             color='blue', fontsize=10, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # SDR 클리핑 정보
    plt.text(0.98, text_y_offset, 
             f'SDR Max Clipped: {sdr_white_clipped_percent:.2f}%\nSDR Min Clipped: {sdr_black_clipped_percent:.2f}%', 
             color='red', fontsize=10, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    '''

    # --- 6. 플롯 설정 ---
    plt.title('Luminance Distribution: Linear Normalized Comparison', fontsize=16)
    plt.xlabel('Normalized Luminance/Pixel Value Range (0.0 = Min, 1.0 = Max)', fontsize=12)
    plt.ylabel('Pixel Density', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(PLOT_OUTPUT_PATH)
    print(f"\n✅ 휘도 분포 플롯이 저장되었습니다: {PLOT_OUTPUT_PATH}")
    # plt.show() # 환경에 따라 주석 처리
    

# --- 7. 메인 실행 블록 (이미지 로드 및 호출) ---
def main():
    print(f"--- 선형 정규화 분포 분석 시작 ---")
    
    try:
        # 1. HDR 이미지 로드
        if not os.path.exists(HDR_FILE_PATH):
            raise FileNotFoundError(f"오류: HDR 입력 파일 '{HDR_FILE_PATH}'을(를) 찾을 수 없습니다.")
        hdr_input = iio.imread(HDR_FILE_PATH) 
        
        # 2. SDR 이미지 로드 (TMO 결과)
        if not os.path.exists(SDR_FILE_PATH):
            raise FileNotFoundError(f"오류: TMO 결과 파일 '{SDR_FILE_PATH}'을(를) 찾을 수 없습니다. TMO 변환 코드를 먼저 실행하여 파일을 생성하세요.")
        sdr_input = iio.imread(SDR_FILE_PATH)
        
        # JPEG 파일은 8비트(0-255)이므로 0.0-1.0 float로 변환
        sdr_input = sdr_input.astype(np.float32) / 255.0
        
        print(f"-> 이미지 로드 완료. HDR 크기: {hdr_input.shape[:2]}, SDR 크기: {sdr_input.shape[:2]}")
        
        # 3. 분포 계산 및 플롯
        plot_linear_normalized_distribution(hdr_input, sdr_input)
        
    except FileNotFoundError as e:
        print(e)
        print("💡 경로와 파일명을 다시 한번 확인해주세요.")
    except Exception as e:
        print(f"❗ 처리 중 오류 발생: {e}")

if __name__ == '__main__':
    main()