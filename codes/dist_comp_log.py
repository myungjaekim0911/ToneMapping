import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os

# --- 파일 경로 설정 (이전과 동일하게 설정) ---
RAW_HDR_FILE_PATH = '~/TM/LDR-HDR-pair_Dataset/HDR/HDR_001.hdr' 
RAW_OUTPUT_DIR = '~/TM/temp_results'
OUTPUT_JPEG_FILENAME = 'output_mean_emor_001.jpg'

# 경로 확장 및 구성
HDR_FILE_PATH = os.path.expanduser(RAW_HDR_FILE_PATH)
OUTPUT_DIR = os.path.expanduser(RAW_OUTPUT_DIR)
SDR_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_JPEG_FILENAME)
PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'log_linear_normalized_luminance_plot.png')


# --- 1. 휘도(Luminance) 계산 함수 ---
def compute_luminance(image_data):
    """
    RGB 데이터로부터 ITU-R BT.709 표준을 기반으로 휘도를 계산합니다.
    """
    # 0.2126R + 0.7152G + 0.722B
    return 0.2126 * image_data[..., 0] + 0.7152 * image_data[..., 1] + 0.0722 * image_data[..., 2]

# --- 2. 분포 분석 및 플롯 함수 (HDR 로그 스케일 적용) ---
def plot_log_linear_normalized_distribution(hdr_data, sdr_data):
    """
    HDR은 로그 정규화, SDR은 선형 스케일을 사용하여 분포를 플롯합니다.
    """
    
    # 픽셀 값 비교를 위한 허용 오차 설정
    HDR_EPSILON = 1e-4  # HDR min/max에 대한 오차
    SDR_EPSILON = 1e-8  # TMO 클리핑(0.0/1.0)에 대한 오차
    EPSILON_LOG = 1e-5  # 로그 계산 시 0 방지
    
    # 1. 휘도 추출 및 평탄화
    L_hdr = compute_luminance(hdr_data).flatten()
    L_sdr = compute_luminance(sdr_data).flatten()
    total_pixels = L_hdr.size
    
    # --- 2. HDR 데이터 분석 및 로그 정규화 ---
    
    # 0 근처 픽셀은 로그 계산에서 제외하고 분석
    L_hdr_positive = L_hdr[L_hdr >= EPSILON_LOG] 
    
    # 원본 HDR 휘도 범위 (클리핑 분석용)
    hdr_min_orig = L_hdr_positive.min() if L_hdr_positive.size > 0 else 0
    hdr_max_orig = L_hdr_positive.max() if L_hdr_positive.size > 0 else 1
    
    # HDR 클리핑 비율 계산 (최대/최소 휘도에 근접한 픽셀)
    # L_hdr_positive 기준으로 min/max 근처 픽셀 카운트
    hdr_white_clipped_count = np.sum(np.isclose(L_hdr, hdr_max_orig, atol=HDR_EPSILON))
    hdr_black_clipped_count = np.sum(np.isclose(L_hdr, hdr_min_orig, atol=HDR_EPSILON))
    
    # 로그 변환 및 정규화
    L_hdr_log = np.log(L_hdr_positive)
    log_min = L_hdr_log.min()
    log_max = L_hdr_log.max()
    log_range = log_max - log_min
    
    # 로그 정규화된 HDR 데이터 (X축 값)
    if log_range < HDR_EPSILON:
         hdr_normalized = np.zeros_like(L_hdr_log)
    else:
         hdr_normalized = (L_hdr_log - log_min) / log_range
    
    hdr_white_clipped_percent = (hdr_white_clipped_count / total_pixels) * 100
    hdr_black_clipped_percent = (hdr_black_clipped_count / total_pixels) * 100
    
    # --- 3. SDR 데이터 분석 (선형) ---
    
    # SDR 클리핑 비율 계산 (정확히 0.0 또는 1.0에 클리핑된 픽셀)
    sdr_white_clipped_count = np.sum(np.isclose(L_sdr, 1.0, atol=SDR_EPSILON))
    sdr_black_clipped_count = np.sum(np.isclose(L_sdr, 0.0, atol=SDR_EPSILON))

    sdr_normalized = L_sdr # SDR은 이미 0~1 선형 스케일
    
    sdr_white_clipped_percent = (sdr_white_clipped_count / total_pixels) * 100
    sdr_black_clipped_percent = (sdr_black_clipped_count / total_pixels) * 100
    
    # --- 4. 히스토그램 계산 및 플롯 ---
    
    # HDR 히스토그램 (로그 정규화된 X축 사용)
    # bins는 X축 범위 (0.0~1.0)에 맞춰 256개
    hdr_hist, bins = np.histogram(hdr_normalized, bins=256, range=(0.0, 1.0), density=True)
    centers = (bins[:-1] + bins[1:]) / 2 # HDR X축
    
    # SDR 히스토그램 (선형 스케일 사용)
    sdr_hist, _ = np.histogram(sdr_normalized, bins=256, range=(0.0, 1.0), density=True)
    # SDR X축은 centers와 동일한 0~1 범위를 사용함
    
    plt.figure(figsize=(12, 7))
    
    # 1. HDR 휘도 분포 플롯
    plt.plot(centers, hdr_hist, 
             label=f'HDR (Input) Distribution (Log Normalized)', 
             color='blue', linewidth=2, alpha=0.7)
    
    # 2. SDR 휘도 분포 플롯
    plt.plot(centers, sdr_hist, 
             label=f'SDR (TMO Output) Distribution (Linear Scale)', 
             color='red', linestyle='--', linewidth=2, alpha=0.9)
    
    # 3. 텍스트 정보 표시
    text_y_offset = plt.ylim()[1] * 0.95
    
    '''
    # HDR 클리핑 정보
    plt.text(0.02, text_y_offset, 
             f'HDR Max Clipped (at {hdr_max_orig:.2f}): {hdr_white_clipped_percent:.2f}%\nHDR Min Clipped (at {hdr_min_orig:.2f}): {hdr_black_clipped_percent:.2f}%', 
             color='blue', fontsize=10, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # SDR 클리핑 정보
    plt.text(0.98, text_y_offset, 
             f'SDR Max Clipped (at 1.0): {sdr_white_clipped_percent:.2f}%\nSDR Min Clipped (at 0.0): {sdr_black_clipped_percent:.2f}%', 
             color='red', fontsize=10, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    '''

    # --- 5. 플롯 설정 ---
    plt.title('Luminance Distribution: HDR Log vs SDR Linear Comparison', fontsize=16)
    plt.xlabel('Normalized Range (0.0 to 1.0)', fontsize=12)
    plt.ylabel('Pixel Density', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(PLOT_OUTPUT_PATH)
    print(f"\n✅ 휘도 분포 플롯이 저장되었습니다: {PLOT_OUTPUT_PATH}")
    # plt.show() # 환경에 따라 주석 처리
    

# --- 6. 메인 실행 블록 ---
def main():
    print(f"--- HDR 로그/SDR 선형 분포 분석 시작 ---")
    
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
        plot_log_linear_normalized_distribution(hdr_input, sdr_input)
        
    except FileNotFoundError as e:
        print(e)
        print("💡 경로와 파일명을 다시 한번 확인해주세요.")
    except Exception as e:
        print(f"❗ 처리 중 오류 발생: {e}")

if __name__ == '__main__':
    main()