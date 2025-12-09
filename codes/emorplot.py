import numpy as np
import matplotlib.pyplot as plt
import os

def parse_emor_data(file_path):
    """
    emor.txt 파일에서 EMoR 모델의 데이터를 파싱합니다.
    E: 입력 휘도 샘플링 포인트 (X축)
    f0: 평균 CRF 곡선 (Y축)
    h: PCA 기저 벡터 (25개)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")

    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    # 라인 내의 숫자 문자열을 파싱하는 헬퍼 함수
    def _parse(lines, tag):
        for line_idx, line in enumerate(lines):
            if line == tag:
                break
        
        s_idx = line_idx + 1
        r = []
        # 데이터는 4개씩 분리되어 있어, 1000/4 = 250줄을 읽습니다.
        for idx in range(s_idx, s_idx + int(1000 / 4)):
            r += lines[idx].split()

        return np.float32(r)

    # 1. 입력 휘도 E (X축) 파싱: "E ="
    E = _parse(lines, 'E =')

    # 2. 평균 응답 f0 (Y축) 파싱: "f0 ="
    f0 = _parse(lines, 'f0 =')

    # 3. PCA 기저 벡터 h 파싱: "h(1)=" 부터 "h(25)=" 까지
    H = np.stack([_parse(lines, 'h(%d)=' % (i + 1)) for i in range(25)], axis=-1)

    return E, f0, H

def plot_emor_crf(E, f0, pca_weights=None):
    """
    EMoR 데이터를 사용하여 CRF 곡선을 재구성하고 플롯합니다.
    pca_weights가 None이거나 모두 0이면 평균 곡선만 플롯됩니다.
    """
    
    # 1. 가중치 설정: 디폴트 (평균 곡선만)
    if pca_weights is None:
        pca_weights = np.zeros(25, dtype=np.float32)

    # 2. PCA를 이용한 CRF 곡선 재구성
    # 재구성된 곡선 = 평균 응답(f0) + (PCA 가중치) * (PCA 기저 벡터 H)
    
    # H의 형태: [1000, 25], pca_weights의 형태: [25]
    weighted_h = np.dot(H, pca_weights) # [1000]

    # 최종 CRF 곡선 (I = f0 + sum(w_k * h_k))
    crf_curve = f0 + weighted_h
    
    # 3. 플롯
    plt.figure(figsize=(10, 6))
    
    # X축: E (입력 휘도), Y축: I (출력 픽셀 값)
    plt.plot(E, crf_curve, label='EMoR Reconstructed CRF', linewidth=2, color='darkred')
    
    # 평균 곡선 레이블 추가
    if np.all(pca_weights == 0):
        plt.plot(E, f0, '--', label='Mean CRF ($\mathbf{f}_0$) Only', linewidth=1, color='gray')
    
    plt.title('Empirical Model of Response (EMoR) - Mean CRF')
    plt.xlabel('Scene Linear Radiance (E)')
    plt.ylabel('LDR Pixel Value (I)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # CRF의 일반적인 S자 형태를 시각적으로 잘 보여주기 위해 로그 스케일을 적용할 수도 있습니다.
    # plt.xscale('log') # 필요하다면 이 줄을 주석 해제하여 X축 로그 스케일 적용

    plt.show()


# --- 실행 ---
try:
    E, f0, H = parse_emor_data('../dataset/emorCurves.txt')
    print("EMoR 데이터 파싱 완료.")
    print(f"입력 휘도(E) 샘플 수: {len(E)}")
    print(f"PCA 기저 벡터(H) 수: {H.shape[1]}")

    # 디폴트 가중치 (모두 0)를 사용하여 평균 CRF 곡선 플롯
    plot_emor_crf(E, f0)
    
except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"데이터 처리 중 오류 발생: {e}")