import numpy as np
import matplotlib.pyplot as plt
import os

def parse_dorf_curves(file_path):
    """
    dorfCurves.txt 파일에서 CRF(I)와 해당 CRF의 역함수(B) 곡선 데이터를 파싱합니다.
    I는 LDR 픽셀 값, B는 해당 픽셀 값에 매핑되는 선형 휘도입니다. (CRF의 역함수, InvCRF)
    """
    # dorfCurves.txt 파일을 읽기 모드로 엽니다.
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 공백을 제거하여 깔끔하게 만듭니다.
        lines = [line.strip() for line in lines]

    # I (LDR 픽셀 값)와 B (선형 휘도 값) 데이터를 추출합니다.
    # 각 곡선은 6줄 간격으로 반복됩니다.
    # I 값은 [idx + 3], B 값은 [idx + 5] 줄에 있습니다.
    
    # I (LDR 픽셀 값) 데이터 추출
    i_str_list = [lines[idx + 3] for idx in range(0, len(lines), 6)]
    # B (선형 휘도 값) 데이터 추출
    b_str_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]

    # 문자열 리스트를 숫자 배열로 변환합니다.
    # 각 줄의 값들을 공백 기준으로 분리하고, float32 형태로 변환합니다.
    i_list = [ele.split() for ele in i_str_list]
    b_list = [ele.split() for ele in b_str_list]

    # numpy 배열로 최종 변환합니다. (shape: [201, 1000])
    # 여기서 I는 LDR 픽셀 값 [0, 1], B는 선형 휘도 값입니다.
    # CRF는 B (입력) -> I (출력) 함수이며, InvCRF는 I (입력) -> B (출력) 함수입니다.
    I_data = np.float32(i_list)  # LDR Pixel Values (y-axis when plotting InvCRF)
    B_data = np.float32(b_list)  # Linear Radiance Values (x-axis when plotting InvCRF)
    
    # 여기서 I_data는 x축에 해당하는 0부터 1까지의 1000개 샘플, 
    # B_data는 y축에 해당하는 0부터 1까지의 1000개 샘플로 보입니다. (B는 InvCRF의 출력)
    # 실제 CRF 데이터셋의 형태에 맞춰 I (x축) -> B (y축) 형태로 반환합니다.

    return I_data, B_data

def plot_invcrf_curves(I_data, B_data):
    """
    파싱된 I (LDR 픽셀 값)와 B (선형 휘도 값) 데이터를 플롯합니다.
    이 데이터는 InvCRF (Inverse CRF) 곡선입니다. 
    InvCRF: LDR Pixel Value (I) -> Scene Linear Radiance (B)
    """
    num_curves = I_data.shape[0]
    
    plt.figure(figsize=(10, 7))
    
    # 201개의 모든 곡선을 플롯합니다.
    for i in range(num_curves):
        # I가 LDR 픽셀 값 [0, 1]에 대한 샘플링 포인트(x축)
        # B가 그에 대응하는 선형 휘도 값(y축)을 나타냅니다.
        # 즉, x축: LDR Pixel (I), y축: Linear Radiance (B)
        plt.plot(I_data[i], B_data[i], alpha=0.3) 

    # plt.plot(I_data[0], B_data[0], alpha=0.3) 
        
    plt.title('Inverse Camera Response Functions (InvCRF)')
    plt.xlabel('LDR Pixel Value (Normalized)')
    plt.ylabel('Scene Linear Radiance (Normalized)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 201개의 곡선이 겹쳐져 있는 시각적 결과를 보여주기 위해 투명도(alpha)를 사용했습니다.
    plt.show()

# --- 실행 부분 ---
# 'dorfCurves.txt' 파일이 이 코드를 실행하는 디렉토리에 있어야 합니다.
file_name = 'dorfCurves.txt'
if os.path.exists(file_name):
    # 1. 데이터 파싱
    I_data, B_data = parse_dorf_curves(file_name)

    # 2. 곡선 개수 확인
    print(f"파싱된 곡선 개수: {I_data.shape[0]}개")
    print(f"샘플링 포인트 개수: {I_data.shape[1]}개")
    
    # 3. 플롯
    plot_invcrf_curves(I_data, B_data)

else:
    print(f"오류: '{file_name}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")