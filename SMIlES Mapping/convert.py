# map_smiles_in_csv.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles
import logging
from tqdm import tqdm

# --- 설정 ---
# 입력 파일과 출력 파일의 이름을 지정합니다.
INPUT_CSV_FILE = 'predictions_melting_point.csv'
OUTPUT_CSV_FILE = 'predictions_with_mapped_smiles.csv'

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_mapped_smiles(smiles: str) -> str:
    """
    SMILES 문자열을 받아 각 heavy atom에 번호를 매긴 Mapped SMILES로 변환합니다.
    """
    # 입력값이 유효한 문자열인지 확인
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"SMILES 파싱 실패: {smiles}")
        return ""
        
    # 각 heavy atom에 1부터 시작하는 번호(map number)를 부여합니다.
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
        
    # 번호가 매겨진 분자 구조를 SMILES 문자열로 변환하여 반환합니다.
    try:
        # canonical=False 옵션은 원자 순서를 그대로 유지하여 Mapped SMILES를 생성합니다.
        return MolToSmiles(mol, canonical=False)
    except Exception as e:
        logging.error(f"MolToSmiles 변환 실패 '{smiles}': {e}")
        return ""

# --- 메인 실행 로직 ---
def main():
    """
    CSV 파일을 읽어 SMILES를 Mapped SMILES로 변환하고 결과를 저장합니다.
    """
    try:
        # 1. CSV 파일 읽기
        df = pd.read_csv(INPUT_CSV_FILE)
        logging.info(f"'{INPUT_CSV_FILE}' 파일을 성공적으로 읽었습니다. 총 {len(df)}개의 데이터.")
    except FileNotFoundError:
        logging.error(f"오류: 입력 파일 '{INPUT_CSV_FILE}'을 찾을 수 없습니다.")
        return

    # 2. 'SMILES' 컬럼에 변환 함수 적용
    # tqdm을 사용하여 진행 상황을 시각적으로 표시합니다.
    tqdm.pandas(desc="SMILES 변환 중")
    # .apply()를 사용하여 'SMILES' 컬럼의 각 값에 함수를 효율적으로 적용합니다.
    df['SMILES'] = df['SMILES'].progress_apply(convert_to_mapped_smiles)

    # 3. 컬럼 이름 변경
    # 변환된 내용에 맞게 컬럼 이름을 'Mapped_SMILES'로 변경합니다.
    df.rename(columns={'SMILES': 'Mapped_SMILES'}, inplace=True)

    # 4. 결과 저장
    # 수정된 데이터프레임을 새로운 CSV 파일로 저장합니다.
    # index=False 옵션은 엑셀에서 불필요한 인덱스 컬럼이 생기는 것을 방지합니다.
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    logging.info(f"작업 완료! 결과가 '{OUTPUT_CSV_FILE}' 파일로 저장되었습니다.")

if __name__ == '__main__':
    main()