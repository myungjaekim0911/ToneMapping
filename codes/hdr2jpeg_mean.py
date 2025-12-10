import numpy as np
import imageio.v3 as iio
import os
# os.path.expanduser와 os.path.join을 사용하여 경로 안정성 확보

# --- 파일 경로 설정 (사용자 요청 반영) ---
# 경로에 ~/와 ../가 포함되어 있으므로, os 모듈을 사용해 정확히 처리합니다.

# 1. 파일 경로를 문자열로 먼저 정의
RAW_HDR_FILE_PATH = '~/TM/LDR-HDR-pair_Dataset/HDR/HDR_001.hdr' 
RAW_EMOR_FILE_PATH = '../dataset/emorCurves.txt'
RAW_OUTPUT_DIR = '~/TM/temp_results'
OUTPUT_JPEG_FILENAME = 'output_mean_emor_001.jpg'

# 2. 경로를 실제 시스템 경로로 확장하고 최종 파일 경로를 구성
HDR_FILE_PATH = os.path.expanduser(RAW_HDR_FILE_PATH)
EMOR_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in locals() else os.getcwd(), RAW_EMOR_FILE_PATH))
OUTPUT_DIR = os.path.expanduser(RAW_OUTPUT_DIR)
OUTPUT_JPEG_PATH = os.path.join(OUTPUT_DIR, OUTPUT_JPEG_FILENAME)


# --- 1. EMoR 데이터 파싱 함수 (변경 없음) ---
def parse_emor_data(file_path):
    """
    emor.txt 파일에서 EMoR 모델의 E (입력 휘도 샘플)와 f0 (평균 CRF 곡선)을 파싱합니다.
    """
    if not os.path.exists(file_path):
        # EMoR 파일 이름이 emorCurves.txt로 변경되었을 수 있으므로, 해당 파일을 다시 확인합니다.
        # 기존 코드에서는 emor.txt를 기준으로 파싱하므로 파일명을 명확히 해야 합니다.
        raise FileNotFoundError(f"오류: EMoR 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")

    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    def _parse(lines, tag):
        # ... (이전 코드와 동일한 파싱 로직) ...
        for line_idx, line in enumerate(lines):
            if line == tag:
                break
        
        s_idx = line_idx + 1
        r = []
        for idx in range(s_idx, s_idx + int(1000 / 4)):
            r += lines[idx].split()

        return np.float32(r)

    # E (입력 휘도)와 f0 (평균 CRF)만 파싱합니다.
    # 주의: 사용자가 파일명을 emorCurves.txt로 언급했지만, 
    # 내부 데이터 태그는 EMoR의 emor.txt 형식을 따르는 것으로 가정합니다.
    E = _parse(lines, 'E =')
    f0 = _parse(lines, 'f0 =')

    return E, f0

# --- 2. Mean EMoR CRF 적용 TMO 함수 (NumPy interp 사용) ---
def apply_mean_emor_tmo(hdr_image, E_samples, mean_crf_f0):
    """
    HDR 이미지에 Mean EMoR CRF 곡선을 적용하여 SDR 이미지로 변환합니다.
    (로그 평균 휘도를 이용한 전역 스케일링 포함)
    """
    # 0. 필수 파라미터 정의
    L_key = 0.18  # 목표 중간 밝기 (Mid-gray point), EMoR E 샘플 범위 내의 값
    epsilon = 1e-5 # 로그 계산의 안정성을 위한 작은 값

    # 1. HDR 이미지의 휘도 (Luminance) 계산
    # ITU-R BT.709 표준에 따른 휘도 공식 (R, G, B 채널 순서 가정)
    L_hdr = 0.2126 * hdr_image[:, :, 0] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 2]
    
    # 2. 로그 평균 휘도 (Log-Average Luminance) 계산
    # log(L + epsilon)의 평균을 낸 후 exp를 적용
    # L_hdr이 0인 경우를 방지하기 위해 epsilon을 더합니다.
    log_L_avg = np.mean(np.log(L_hdr + epsilon))
    L_avg = np.exp(log_L_avg)

    # 3. 스케일 팩터 계산 (Sc = L_key / L_avg)
    # 이미지의 평균 밝기를 L_key (0.18)로 매핑하기 위한 스케일 팩터
    scale_factor = L_key / L_avg
    
    # 4. 각 채널에 스케일 팩터 적용 (Normalization)
    # L_scene_scaled = L_scene * scale_factor
    hdr_scaled = hdr_image * scale_factor
    
    sdr_image = np.zeros_like(hdr_image, dtype=np.float32)

    # 5. 스케일링된 이미지에 CRF 곡선 적용
    for i in range(hdr_image.shape[2]): # R, G, B 채널 반복
        # np.interp를 사용하여 보간 수행
        sdr_image[:, :, i] = np.interp(
            x=hdr_scaled[:, :, i], 
            xp=E_samples, 
            fp=mean_crf_f0,
        )

    # 최종 출력은 0.0 ~ 1.0 범위로 클리핑
    sdr_image = np.clip(sdr_image, 0.0, 1.0)
    
    return sdr_image

# --- 3. SDR 이미지를 8비트 JPEG로 저장하는 함수 ---
def save_sdr_jpeg(sdr_image, output_filename):
    """
    SDR float 이미지를 8비트 JPEG 파일로 저장합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_8bit = (sdr_image * 255).astype(np.uint8)
    
    iio.imwrite(output_filename, image_8bit, quality=95)
    print(f"✅ 변환된 JPEG 파일이 저장되었습니다: {output_filename}")


# --- 4. 메인 실행 블록 ---
def main():
    print(f"--- 파일 경로 설정 확인 ---")
    print(f"HDR 입력 파일: {HDR_FILE_PATH}")
    print(f"EMoR 데이터 파일: {EMOR_FILE_PATH}")
    print(f"JPEG 출력 경로: {OUTPUT_JPEG_PATH}")
    print("--------------------------")
    
    try:
        # 1. EMoR 데이터 로드 및 파싱
        E_samples, mean_crf_f0 = parse_emor_data(EMOR_FILE_PATH)
        print("1. EMoR Mean CRF 곡선 로드 완료.")

        # 2. HDR 이미지 로드
        if not os.path.exists(HDR_FILE_PATH):
            raise FileNotFoundError(f"오류: HDR 입력 파일 '{HDR_FILE_PATH}'을(를) 찾을 수 없습니다.")

        hdr_input = iio.imread(HDR_FILE_PATH) 
        
        print(f"2. HDR 이미지 로드 완료. 크기: {hdr_input.shape[:2]}")

        # 3. TMO 적용
        sdr_output = apply_mean_emor_tmo(hdr_input, E_samples, mean_crf_f0)
        print("3. Mean EMoR TMO 적용 완료.")

        # 4. JPEG 저장
        save_sdr_jpeg(sdr_output, OUTPUT_JPEG_PATH)
        
    except FileNotFoundError as e:
        print(e)
        print("💡 파일을 찾을 수 없습니다. 경로와 파일명을 다시 한번 확인해주세요.")
    except Exception as e:
        print(f"❗ 처리 중 오류 발생: {e}")

if __name__ == '__main__':
    main()