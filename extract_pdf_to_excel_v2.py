import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import pdfplumber
from openpyxl.styles import Alignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =========================================================
# 1. 텍스트 정제
# =========================================================

def clean_text(text: str) -> str:
    """기본 텍스트 정제"""
    if not text:
        return ''
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_text(text: str) -> str:
    """정규화된 텍스트"""
    return clean_text(text)


# =========================================================
# 2. TEXT 전용: 참조문장 & 라벨 제거 (강화)
# =========================================================

def remove_reference_phrases(text: str) -> str:
    """[그림 x-y], [표 x-y] 참조문장 삭제"""
    if not text:
        return ''
    
    # "[표/그림 x-y]와 같이", "[표/그림 x-y]에 제시된" 등
    text = re.sub(r'\[(?:그림|표|Figure|Table)\s+\d+-\d+\][와의이로을를으].*?(?=[.。！？?!다습니다합니다\n]|$)', '', text)
    text = re.sub(r'(?:그림|표|Figure|Table)\s+\d+-\d+[에서를].*?(?=[.。！？?!다습니다합니다\n]|$)', '', text)
    text = re.sub(r'\[(?:그림|표|Figure|Table)\s+\d+-\d+\]', '', text)
    
    return clean_text(text)


def is_diagram_label_line(text: str) -> bool:
    """도면/부속재 라벨 조각 판별 (강화)"""
    text = text.strip()
    
    # 너무 짧음
    if len(text) > 30:
        return False
    
    # 순수 숫자/기호
    if re.match(r'^[\d\.\(\)×÷\-\+\/]+$', text):
        return True
    
    # Ø, X로 시작하는 부재기호
    if re.match(r'^[Ø×]\d+', text):
        return True
    
    # 숫자 X 숫자 (예: 6 X 48)
    if re.match(r'^\d+\s*[Xx×]\s*\d+', text):
        return True
    
    # 단위만
    if re.match(r'^(mm|cm|m|kgf|kN|N|%|도|분|초)$', text):
        return True
    
    # 기호/숫자 비율 높음
    symbol_count = sum(1 for c in text if c in 'Ø×÷±∓=≠≤≥<>()[]{}' or c.isdigit())
    if len(text) > 5 and symbol_count / len(text) > 0.6:
        return True
    
    # 부재/조건 라벨 (H-100, B-200, GL+0.5)
    if re.match(r'^[A-Z]{1,2}[\-\+][\d\.]+', text):
        return True
    
    # 치수 단위 조합 (예: "100mm")
    if re.match(r'^\d+(mm|cm|m|kgf|kN)$', text):
        return True
    
    return False


def remove_diagram_labels(text: str) -> str:
    """도면/표 내부 라벨 제거 (강화)"""
    if not text:
        return ''
    
    text = text.strip()
    
    # 라벨 라인 전체 제거
    if is_diagram_label_line(text):
        return ''
    
    # 특정 단어 라벨
    diagram_keywords = [
        '건축주', '기둥 중심선', '대지 경계', '귀 규준틀', '선 중심', '벽선', '기준점',
        '줄쳐보기', '레벨 조정', '높이 조정', '수직 조정', '수평 조정', '위치 조정',
        '재료', '두께', '폭', '높이', '깊이', '간격', '거리', '각도',
    ]
    
    for kw in diagram_keywords:
        if text == kw:
            return ''
        if re.match(rf'^{re.escape(kw)}[:\s:：]*[\d\.\-\/]+.*$', text):
            return ''
    
    return text


def remove_formula_lines(text: str) -> str:
    """식/공식 라인 제거"""
    if not text:
        return ''
    
    text = text.strip()
    
    # 등식
    if re.match(r'^[A-Z가-힣]\s*[=×÷±∓]+', text):
        return ''
    
    # 설명
    if re.match(r'^(?:여기서|단|단\s+|where)\s+[A-Z가-힣].*?:', text):
        return ''
    
    # 순수 계산식
    if re.match(r'^[\d()\s\+\-\*/\.]+$', text):
        return ''
    
    return text


# =========================================================
# 3. TEXT 전용: 문장 끝 감지 (소수점 무시)
# =========================================================

def is_korean_sentence_end(text: str, pos: int) -> bool:
    """
    한국어 종결형으로 끝나는지 확인
    소수점(1.8, 2.3.1) 무시
    """
    if pos < 0 or pos >= len(text):
        return False
    
    char = text[pos]
    
    # 마침표 체크
    if char == '.':
        # 앞이 숫자고 뒤가 숫자면 소수점
        if pos > 0 and text[pos-1].isdigit():
            if pos + 1 < len(text) and (text[pos+1].isdigit() or text[pos+1] == ' '):
                return False
        
        # 앞이 '다'면 종결형 마침표
        if pos > 0 and text[pos-1] == '다':
            return True
        
        # 앞이 '다' 또는 다른 종결형
        if pos >= 2:
            two_before = text[pos-2:pos]
            if two_before in ['습니다', '합니다', '세요', '십시오']:
                return True
    
    # 물음표, 느낌표
    if char in '?？!！':
        return True
    
    return False


# =========================================================
# 4. TEXT 전용: 문장 분할 (개선)
# =========================================================

def split_into_complete_sentences_v2(text: str) -> List[str]:
    """
    완결 문장 기준 분할
    - 소수점 무시
    - 한국어 종결형 우선
    - 개행으로 잘린 문장 복구
    """
    if not text:
        return []
    
    # Step 1: 라인별 정제
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 라벨 제거
        line = remove_diagram_labels(line)
        line = remove_formula_lines(line)
        
        if line and len(line) >= 4:
            lines.append(line)
    
    if not lines:
        return []
    
    # Step 2: 라인 이어붙이기 (개행 제거)
    text = ' '.join(lines)
    
    # Step 3: 한국어 종결형 기준 분할
    sentences = []
    current = []
    i = 0
    
    while i < len(text):
        char = text[i]
        current.append(char)
        
        # 문장 끝 체크
        if is_korean_sentence_end(text, i):
            # 다음 문자 확인
            if i + 1 < len(text):
                next_char = text[i + 1]
                
                # 공백 후 글자면 분리
                if next_char == ' ':
                    if i + 2 < len(text):
                        after_space = text[i + 2]
                        if re.match(r'[가-힣A-Z0-9①-⑳\(]', after_space):
                            sent = ''.join(current).strip()
                            if _validate_text_sentence(sent):
                                sentences.append(sent)
                            current = []
                            i += 1  # 공백 건너뛰기
                # 공백 없이 바로 글자면 분리
                elif re.match(r'[가-힣A-Z0-9①-⑳\(]', next_char):
                    sent = ''.join(current).strip()
                    if _validate_text_sentence(sent):
                        sentences.append(sent)
                    current = []
            else:
                # 텍스트 끝
                sent = ''.join(current).strip()
                if _validate_text_sentence(sent):
                    sentences.append(sent)
                current = []
        
        i += 1
    
    # 남은 텍스트
    if current:
        sent = ''.join(current).strip()
        if _validate_text_sentence(sent):
            sentences.append(sent)
    
    return sentences


def _is_fragment_line(text: str) -> bool:
    """불완전한 토막 문장 판별"""
    text = text.strip()
    
    # 너무 짧음
    if len(text) < 8:
        return True
    
    # 종결부-only
    if re.match(r'^(한다|있다|된다|이다|설치한다|사용한다|행한다|하고|위해|될수|되고)\.?$', text):
        return True
    
    # 조사/접사-only
    if re.match(r'^(와 같이|에 제시된|의 오른쪽|의 왼쪽|도|도록|하도록|야 한다)\.?$', text):
        return True
    
    # 숫자 fragment
    if re.match(r'^[\d\.\-\+\/]+[a-zA-Z%가-힣]*\.?$', text) and len(text) < 15:
        return True
    
    return False


def _validate_text_sentence(text: str) -> bool:
    """완결 문장 검증"""
    if not text:
        return False
    
    text = text.strip()
    
    # 최소 길이
    if len(text) < 10:
        return False
    
    # 불완전한 토막
    if _is_fragment_line(text):
        return False
    
    # 도면 라벨
    if is_diagram_label_line(text):
        return False
    
    # 종결형 또는 의문형 포함
    has_ending = re.search(r'(한다|습니다|합니다|있다|된다|이다|세요|십시오|[\?？!！])\.?$', text)
    if not has_ending:
        return False
    
    return True


# =========================================================
# 5. TEXT 전용: 문장 복원 & 병합
# =========================================================

def merge_fragmented_sentences(sentences: List[str]) -> List[str]:
    """
    파편화된 문장 복원
    - 종결부-only는 앞과 병합
    - 숫자/단위에서 잘린 것은 다음과 병합
    - heading-only는 다음 1개와만 병합
    """
    if not sentences:
        return []
    
    merged = []
    i = 0
    
    while i < len(sentences):
        sent = sentences[i].strip()
        
        # 제목만 있는 라인이면 다음 1개와만 병합
        if _is_heading_only(sent) and i + 1 < len(sentences):
            merged_sent = sent + ' ' + sentences[i + 1].strip()
            merged.append(merged_sent)
            i += 2
            continue
        
        # 앞 문장이 숫자/단위에서 잘렸다면 이번 문장과 병합
        if merged and _ends_with_number_or_unit(merged[-1]):
            merged[-1] = merged[-1] + ' ' + sent
            i += 1
            continue
        
        # 현재 문장이 매우 짧고 종결부-only면 앞과 병합
        if len(sent) < 15 and _is_fragment_line(sent) and merged:
            merged[-1] = merged[-1] + ' ' + sent
            i += 1
            continue
        
        merged.append(sent)
        i += 1
    
    return merged


def _is_heading_only(text: str) -> bool:
    """제목/번호만 있는지 판별"""
    text = text.strip()
    
    # 순수 번호
    if re.match(r'^[①-⑳⑴-⑽]\s*$', text):
        return True
    if re.match(r'^\(?\d+\)\s*$', text):
        return True
    
    # 번호 + 짧은 제목 (20자 미만)
    if re.match(r'^[①-⑳⑴-⑽]\s+[가-힣A-Za-z\s]+$', text) and len(text) < 20:
        return True
    if re.match(r'^\(?\d+\)\s+[가-힣A-Za-z\s]+$', text) and len(text) < 20:
        return True
    
    return False


def _ends_with_number_or_unit(text: str) -> bool:
    """숫자나 단위로 끝나는지 확인"""
    text = text.strip()
    
    # 숫자로 끝남
    if re.search(r'[\d\.]+$', text):
        return True
    
    # 단위로 끝남
    if re.search(r'(m|mm|cm|kgf|kN|N|%|도|분|초)$', text):
        return True
    
    # 조사로 끝남 (다음 단어가 올 가능성)
    if re.search(r'(는|은|를|을|와|과|에|에서|로)$', text):
        return True
    
    return False


# =========================================================
# 6. TEXT 전용: 최종 문장 구축
# =========================================================

def build_complete_text_sentences_v2(subsections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    완결 문장 단위 text row 생성 (개선)
    - 참조문장 제거
    - 도면 라벨 제거
    - 문장 복원
    - 병합
    - 검증
    """
    sentences_list = []
    sent_id = 1
    
    for sub in subsections:
        raw_text = sub.get('text', '').strip()
        if not raw_text:
            continue
        
        # Step 1: 참조문장 제거
        text = remove_reference_phrases(raw_text)
        if not text or len(text) < 10:
            continue
        
        # Step 2: 문장 분할 (개선)
        sentences = split_into_complete_sentences_v2(text)
        if not sentences:
            continue
        
        # Step 3: 문장 복원 & 병합
        final_sentences = merge_fragmented_sentences(sentences)
        if not final_sentences:
            continue
        
        # Step 4: 각 문장을 1 row로
        for sent in final_sentences:
            sent = sent.strip()
            
            # 최종 검증
            if not _validate_text_sentence(sent):
                continue
            
            sentences_list.append({
                'sent_id': f'S_{sent_id:05d}',
                'subsection': sub['subsection'],
                'subsection_title': sub['subsection_title'],
                'section': sub['section'],
                'section_title': sub['section_title'],
                'chapter': sub['chapter'],
                'chapter_title': sub['chapter_title'],
                'page_number': sub['page_number'],
                'position': sub['position'],
                'top': sub['top'],
                'text': sent,
            })
            sent_id += 1
    
    return sentences_list


# =========================================================
# 7. PDF 라인 추출 (기존)
# =========================================================

def _is_two_column_page(words: List[Dict[str, Any]], page_width: float) -> bool:
    """2단 레이아웃 감지"""
    if not words:
        return False
    mid_lo, mid_hi = page_width * 0.40, page_width * 0.60
    mid_words = [w for w in words if mid_lo <= (w['x0'] + w['x1']) / 2 <= mid_hi]
    ratio = len(mid_words) / len(words) if words else 0
    return ratio < 0.08


def _estimate_col_split(page, words: List[Dict[str, Any]]) -> float:
    """단 분리선 추정"""
    width = page.width
    if not _is_two_column_page(words, width):
        return width
    
    x_mids = sorted(set(int((w['x0'] + w['x1']) / 2) for w in words))
    if len(x_mids) > 2:
        gaps = [(x_mids[i + 1] - x_mids[i], x_mids[i]) for i in range(len(x_mids) - 1)]
        max_gap_val, max_gap_x = max(gaps, key=lambda g: g[0])
        if max_gap_val >= 20 and width * 0.35 <= max_gap_x <= width * 0.65:
            return max_gap_x + max_gap_val / 2
    return width / 2


def _group_words_to_lines(words: List[Dict[str, Any]], y_tol: float = 3.0) -> List[Dict[str, Any]]:
    """단어를 라인으로 그룹화"""
    if not words:
        return []
    
    words = sorted(words, key=lambda w: (round(w['top'] / y_tol) * y_tol, w['x0']))
    lines = []
    current = []
    current_top = None
    
    for w in words:
        if current_top is None:
            current = [w]
            current_top = w['top']
        elif abs(w['top'] - current_top) <= y_tol:
            current.append(w)
        else:
            if current:
                current = sorted(current, key=lambda x: x['x0'])
                text = clean_text(' '.join(x['text'] for x in current))
                if text:
                    lines.append({
                        'text': text,
                        'x0': min(x['x0'] for x in current),
                        'x1': max(x['x1'] for x in current),
                        'top': min(x['top'] for x in current),
                        'bottom': max(x['bottom'] for x in current),
                    })
            current = [w]
            current_top = w['top']
    
    if current:
        current = sorted(current, key=lambda x: x['x0'])
        text = clean_text(' '.join(x['text'] for x in current))
        if text:
            lines.append({
                'text': text,
                'x0': min(x['x0'] for x in current),
                'x1': max(x['x1'] for x in current),
                'top': min(x['top'] for x in current),
                'bottom': max(x['bottom'] for x in current),
            })
    
    return lines


def extract_lines_with_layout(pdf_path: str) -> List[Dict[str, Any]]:
    """PDF에서 라인 추출"""
    lines_out = []
    footer_margin = 60
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_height = page.height
            page_width = page.width
            words = page.extract_words() or []
            words = [w for w in words if w['bottom'] <= page_height - footer_margin]
            
            if not words:
                continue
            
            col_split = _estimate_col_split(page, words)
            columns = [('left', 0, page_width)] if col_split >= page_width else [
                ('left', 0, col_split), ('right', col_split, page_width)
            ]
            
            for position, x_min, x_max in columns:
                part_words = [w for w in words if x_min <= w['x0'] and w['x1'] <= x_max + 1]
                for line in _group_words_to_lines(part_words):
                    if line['text']:
                        lines_out.append({
                            'page_num': page_num,
                            'position': position,
                            'text': line['text'],
                            'top': line['top'],
                            'bottom': line['bottom'],
                            'x0': line['x0'],
                            'x1': line['x1'],
                        })
    
    return lines_out


# =========================================================
# 8. 문서 구조 추출 (기존)
# =========================================================

def extract_document_structure(lines: List[Dict[str, Any]], pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """문서 구조 추출"""
    chapter_no = os.path.splitext(os.path.basename(pdf_path))[0]
    m = re.match(r'(\d+)\.(.+)', chapter_no)
    if m:
        chapter_no, chapter_title = m.group(1), m.group(2)
    else:
        chapter_title = chapter_no
    
    structure = {'sections': [], 'subsections': []}
    current_section = None
    seen_sections = set()
    seen_subsections = set()
    
    for line in sorted(lines, key=lambda l: (l['page_num'], 0 if l['position'] == 'left' else 1, l['top'])):
        text = line['text'].strip()
        if not text:
            continue
        
        # 소절
        sub_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.+)$', text)
        if sub_match and re.search(r'[가-힣A-Za-z]', sub_match.group(2)):
            subsection = sub_match.group(1)
            parts = subsection.split('.')
            if chapter_no and parts[0] != chapter_no:
                pass
            elif subsection not in seen_subsections:
                if all(p.isdigit() and 0 < int(p) < 99 for p in parts):
                    title = sub_match.group(2)[:50]
                    structure['subsections'].append({
                        'chapter': chapter_no,
                        'chapter_title': chapter_title,
                        'section': current_section['section'] if current_section else '',
                        'section_title': current_section['section_title'] if current_section else '',
                        'subsection': subsection,
                        'subsection_title': title,
                        'page_number': line['page_num'],
                        'position': line['position'],
                        'top': line['top'],
                        'text': '',
                    })
                    seen_subsections.add(subsection)
            continue
        
        # 절
        sec_match = re.match(r'^(\d+\.\d+)\s+(.+)$', text)
        if sec_match and re.search(r'[가-힣A-Za-z]', sec_match.group(2)):
            section = sec_match.group(1)
            parts = section.split('.')
            if chapter_no and parts[0] != chapter_no:
                pass
            elif section not in seen_sections:
                if all(p.isdigit() and 0 < int(p) < 99 for p in parts):
                    title = sec_match.group(2)[:50]
                    current_section = {
                        'chapter': chapter_no,
                        'chapter_title': chapter_title,
                        'section': section,
                        'section_title': title,
                        'page_number': line['page_num'],
                        'position': line['position'],
                        'top': line['top'],
                    }
                    structure['sections'].append(current_section)
                    seen_sections.add(section)
    
    return structure


def fill_subsection_texts(structure: Dict[str, List[Dict[str, Any]]], lines: List[Dict[str, Any]]) -> None:
    """각 소절에 원문 채우기"""
    def line_sort_key(l):
        return (l['page_num'], 0 if l.get('position', 'left') == 'left' else 1, l.get('top', 0))
    
    def sub_sort_key(s):
        return (s['page_number'], 0 if s.get('position', 'left') == 'left' else 1, s.get('top', 0))
    
    subsections = sorted(structure['subsections'], key=sub_sort_key)
    sorted_lines = sorted(lines, key=line_sort_key)
    
    for idx, sub in enumerate(subsections):
        sub['text'] = ''
        sub_key = sub_sort_key(sub)
        next_key = sub_sort_key(subsections[idx + 1]) if idx + 1 < len(subsections) else (10**9, 10**9, 10**9)
        
        texts = []
        for line in sorted_lines:
            lk = line_sort_key(line)
            if lk <= sub_key:
                continue
            if lk >= next_key:
                break
            
            txt = line['text'].strip()
            if not txt:
                continue
            
            if re.match(rf'^{re.escape(sub["subsection"])}\s+', txt):
                continue
            
            if re.match(r'^\d+\.\d+\.\d+\s+', txt) or re.match(r'^\d+\.\d+\s+', txt):
                continue
            
            texts.append(txt)
        
        sub['text'] = '\n'.join(texts)


# =========================================================
# 9. 표 추출 (기존 - 수정 없음)
# =========================================================

def _is_junk_table(rows: List[List]) -> bool:
    """쓰레기 표 필터링"""
    if not rows or len(rows) < 2:
        return True
    
    if not rows[0] or len(rows[0]) < 2:
        return True
    
    all_text = ' '.join(str(c) for row in rows for c in row if c)
    if not all_text.strip():
        return True
    
    dot_chars = sum(1 for c in all_text if c in '…·.·⋯')
    if len(all_text) > 0 and dot_chars / len(all_text) > 0.3:
        return True
    
    if 'CHAPTER' in all_text and len(all_text) < 200:
        return True
    
    total_cells = sum(1 for row in rows for c in row if c and str(c).strip())
    if total_cells <= 2:
        return True
    
    return False


def _merge_none_cells(rows: List[List]) -> List[List]:
    """병합 셀 처리"""
    if not rows:
        return rows
    
    filled = [list(row) for row in rows]
    num_cols = max(len(r) for r in filled) if filled else 0
    
    for row in filled:
        while len(row) < num_cols:
            row.append('')
    
    for r_idx, row in enumerate(filled):
        for c_idx in range(1, len(row)):
            if row[c_idx] is None:
                row[c_idx] = row[c_idx - 1] if row[c_idx - 1] is not None else ''
    
    for c_idx in range(num_cols):
        for r_idx in range(1, len(filled)):
            if c_idx < len(filled[r_idx]) and (filled[r_idx][c_idx] is None or filled[r_idx][c_idx] == ''):
                if c_idx < len(filled[r_idx - 1]):
                    filled[r_idx][c_idx] = filled[r_idx - 1][c_idx]
    
    deduped = []
    for row in filled:
        if not deduped or row != deduped[-1]:
            deduped.append(row)
    
    return deduped


def _position_from_x(x_center: float, col_split: float, page_width: float) -> str:
    """position 결정"""
    if col_split >= page_width:
        return 'left'
    return 'left' if x_center < col_split else 'right'


def _line_matches_table_caption(text: str) -> bool:
    """표 캡션 패턴"""
    return bool(re.match(r'^\[표\s*\d+-\d+\]', text.strip()))


def extract_table_blocks_from_pdf(pdf_path: str, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """표 추출 (기존과 동일)"""
    caption_lines = [l for l in lines if _line_matches_table_caption(l['text'])]
    caption_by_page = defaultdict(list)
    for cap in caption_lines:
        caption_by_page[cap['page_num']].append(cap)
    
    tables_out = []
    
    col_splits = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words() or []
            col_splits[page_idx] = _estimate_col_split(page, words)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_caps = caption_by_page.get(page_idx, [])
            if not page_caps:
                continue
            
            col_split = col_splits[page_idx]
            
            found = []
            for strategy in [
                {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines',
                 'intersection_tolerance': 5, 'snap_tolerance': 3, 'join_tolerance': 3,
                 'edge_min_length': 10, 'min_words_vertical': 2, 'min_words_horizontal': 1},
                {'vertical_strategy': 'lines', 'horizontal_strategy': 'text',
                 'intersection_tolerance': 5, 'snap_tolerance': 3},
                {'vertical_strategy': 'text', 'horizontal_strategy': 'text',
                 'intersection_tolerance': 5, 'snap_tolerance': 3},
            ]:
                try:
                    found = page.find_tables(table_settings=strategy) or []
                    if found:
                        break
                except Exception:
                    continue
            
            for tbl_idx, tbl in enumerate(found, start=1):
                x0, top, x1, bottom = tbl.bbox
                position = _position_from_x((x0 + x1) / 2, col_split, page.width)
                
                raw_rows = tbl.extract() or []
                raw_rows = _merge_none_cells(raw_rows)
                
                if len(raw_rows) < 2 or (raw_rows and len(raw_rows[0]) < 2):
                    continue
                
                if _is_junk_table(raw_rows):
                    logger.info(f'[표 오탐 필터] p{page_idx} tbl{tbl_idx} 제거')
                    continue
                
                cleaned_rows = []
                for row in raw_rows:
                    row = [clean_text(str(c)) if c is not None else '' for c in row]
                    if any(cell for cell in row):
                        cleaned_rows.append(row)
                
                if not cleaned_rows or len(cleaned_rows) < 2:
                    continue
                
                chosen_cap = None
                best_score = None
                for cap in page_caps:
                    cap_center_x = (cap.get('x0', 0) + cap.get('x1', cap.get('x0', 0))) / 2
                    cap_position = _position_from_x(cap_center_x, col_split, page.width)
                    
                    if cap_position != position:
                        continue
                    
                    cap_top = cap.get('top', 0)
                    if cap_top >= top:
                        continue
                    
                    distance = top - cap_top
                    if distance > 100:
                        continue
                    
                    if best_score is None or distance < best_score:
                        chosen_cap = cap
                        best_score = distance
                
                if not chosen_cap:
                    logger.info(f'[표 캡션 없음] p{page_idx} tbl{tbl_idx} 제거')
                    continue
                
                title = chosen_cap['text']
                content_text_raw = '\n'.join(['|'.join(row) for row in cleaned_rows])
                
                tables_out.append({
                    'title': title,
                    'type': 'table',
                    'page_number': page_idx,
                    'position': position,
                    'top': top,
                    'bottom': bottom,
                    'table_rows': cleaned_rows,
                    'content_text_raw': content_text_raw,
                    'content_text_normalized': clean_text(content_text_raw),
                    'table_rows_json': json.dumps(cleaned_rows, ensure_ascii=False),
                })
                
                logger.info(f'✅ p{page_idx} - 표: {title[:30]}... ({len(cleaned_rows)}행 x {len(cleaned_rows[0])}열)')
    
    return sorted(tables_out, key=lambda x: (x['page_number'], x.get('position', ''), x.get('top', 0)))


# =========================================================
# 10. 그림 추출 (기존 - 수정 없음)
# =========================================================

def split_figure_captions(text: str) -> List[str]:
    """여러 캡션 분리"""
    if not text:
        return []
    
    text = clean_text(text)
    matches = list(re.finditer(r'\[그림\s*\d+-\d+\]', text))
    
    if not matches:
        return []
    
    if len(matches) == 1:
        caption = _clean_figure_caption(text)
        return [caption] if caption and len(caption) >= 5 else []
    
    captions = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        cap_text = text[start:end].strip()
        
        caption = _clean_figure_caption(cap_text)
        if caption and len(caption) >= 5:
            captions.append(caption)
    
    return captions


def _clean_figure_caption(text: str) -> str:
    """캡션 정제"""
    text = clean_text(text)
    
    m = re.search(r'\[그림\s*\d+-\d+\]', text)
    if m:
        text = text[m.start():]
    
    text = re.sub(r'\s*출처\s*[:：]\s*\S+', '', text).strip()
    text = re.sub(r'\s*[Ss]ource\s*[:：]\s*\S+', '', text).strip()
    
    if re.search(r'\[그림\s*\d+-\d+\][와의이로을를으]', text):
        return ''
    if re.search(r'\[그림\s*\d+-\d+\](?:의 오른쪽|의 왼쪽|과 같이|에 제시된|과 함께|을 참고)', text):
        return ''
    
    return text


def extract_figure_blocks_v2(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """그림 캡션 추출 (기존과 동일)"""
    figures = []
    fig_id = 1
    
    lines_by_col = defaultdict(list)
    for line in lines:
        lines_by_col[(line['page_num'], line.get('position', ''))].append(line)
    
    for key, col_lines in lines_by_col.items():
        col_lines = sorted(col_lines, key=lambda x: x.get('top', 0))
        
        for idx, line in enumerate(col_lines):
            txt = line['text'].strip()
            
            if not re.match(r'^\[그림\s*\d+-\d+\]', txt):
                continue
            
            captions = split_figure_captions(txt)
            
            if not captions:
                continue
            
            prev_context = []
            j = idx - 1
            while j >= 0 and len(prev_context) < 3:
                prev_txt = col_lines[j]['text'].strip()
                if (prev_txt
                        and not re.match(r'^\[그림\s*\d+-\d+\]', prev_txt)
                        and not re.match(r'^\[표\s*\d+-\d+\]', prev_txt)
                        and not re.search(r'출처\s*[:：]', prev_txt)):
                    prev_context.append(prev_txt)
                j -= 1
            prev_context.reverse()
            
            page_num = line['page_num']
            cap_top = line.get('top', 0)
            cap_pos = line.get('position', '')
            
            for caption in captions:
                figures.append({
                    'title': caption,
                    'type': 'figure',
                    'page_number': page_num,
                    'position': cap_pos,
                    'top': cap_top,
                    'caption_text': caption,
                    'context_before_raw': '\n'.join(prev_context),
                })
                
                logger.info(f'✅ p{page_num} - 그림: {caption[:30]}...')
    
    return figures


# =========================================================
# 11. 블록 매핑 (기존 - 수정 없음)
# =========================================================

def map_blocks_to_subsections(structure: Dict[str, List[Dict[str, Any]]], blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """블록 매핑"""
    def pos_order(p: str) -> int:
        return 0 if p == 'left' else 1
    
    subsections = structure['subsections']
    global_subs = sorted(subsections, key=lambda s: (s['page_number'], pos_order(s.get('position', 'left')), s.get('top', 0)))
    
    results = []
    for info in blocks:
        page = info['page_number']
        pos = info.get('position', '')
        y = info.get('top', 0)
        blk_key = (page, pos_order(pos), y)
        
        matched = None
        for sub in global_subs:
            sub_key = (sub['page_number'], pos_order(sub.get('position', 'left')), sub.get('top', 0))
            if sub_key <= blk_key:
                matched = sub
            else:
                break
        
        if not matched:
            for sub in reversed(global_subs):
                if sub['page_number'] == page and sub['position'] == pos:
                    matched = sub
                    break
        
        if matched:
            results.append({
                'chapter': matched.get('chapter', ''),
                'chapter_title': matched.get('chapter_title', ''),
                'section': matched.get('section', ''),
                'section_title': matched.get('section_title', ''),
                'subsection': matched.get('subsection', ''),
                'subsection_title': matched.get('subsection_title', ''),
                'page_number': info['page_number'],
                'position': info.get('position', ''),
                'top': info.get('top', 0),
                'title': info['title'],
                'type': info['type'],
                'table_rows': info.get('table_rows', []),
                'table_rows_json': info.get('table_rows_json', ''),
                'content_text_raw': info.get('content_text_raw', ''),
                'content_text_normalized': info.get('content_text_normalized', ''),
                'caption_text': info.get('caption_text', ''),
                'context_before_raw': info.get('context_before_raw', ''),
            })
        else:
            results.append({
                'chapter': '',
                'chapter_title': '',
                'section': '',
                'section_title': '',
                'subsection': '',
                'subsection_title': '',
                'page_number': info['page_number'],
                'position': info.get('position', ''),
                'top': info.get('top', 0),
                'title': info['title'],
                'type': info['type'],
                'table_rows': info.get('table_rows', []),
                'table_rows_json': info.get('table_rows_json', ''),
                'content_text_raw': info.get('content_text_raw', ''),
                'content_text_normalized': info.get('content_text_normalized', ''),
                'caption_text': info.get('caption_text', ''),
                'context_before_raw': info.get('context_before_raw', ''),
            })
    
    return results


# =========================================================
# 12. 문제 유형 분류
# =========================================================

def classify_question_type(text: str) -> str:
    """문제 유형 분류"""
    t = text.strip()
    if not t:
        return '특징'
    
    if re.search(r'(을 말한다|를 말한다|의미한다|정의)', t):
        return '정의'
    if re.search(r'\d', t) and re.search(r'(m|mm|cm|kgf|kW|%|이상|이하|강도|높이|폭|간격|강도)', t):
        return '수치'
    if re.search(r'(순서|먼저|이후|다음의|후에|역순|절차)', t):
        return '순서'
    if re.search(r'(분류|종류|구분|구성|나뉜|체계)', t):
        return '분류'
    if re.search(r'(장점|유리|절감|향상|효과)', t):
        return '장점'
    if re.search(r'(단점|문제|불리|주의|제한)', t):
        return '단점'
    
    return '특징'


# =========================================================
# 13. Generation 행 생성
# =========================================================

def build_generation_rows(text_sentences: List[Dict[str, Any]], 
                          mapped_tables: List[Dict[str, Any]],
                          source_file: str) -> pd.DataFrame:
    """Generation_Input 시트"""
    rows = []
    gen_idx = 1
    
    # Text sentences
    for sent in text_sentences:
        text = sent.get('text', '').strip()
        if not text:
            continue
        
        qtype = classify_question_type(text)
        
        rows.append({
            'gen_id': f'GEN_{gen_idx:04d}',
            'source_row_id': sent.get('sent_id', ''),
            'use_for_generation_yn': 'Y',
            'source_type': 'text',
            'page_no': sent.get('page_number', ''),
            'chapter': sent.get('chapter', ''),
            'chapter_title': sent.get('chapter_title', ''),
            'section': sent.get('section', ''),
            'section_title': sent.get('section_title', ''),
            'subsection': sent.get('subsection', ''),
            'subsection_title': sent.get('subsection_title', ''),
            'item_title': sent.get('subsection_title', ''),
            'source_text': text,
            'source_text_raw': text,
            'source_text_normalized': normalize_text(text),
            'question_type_label': qtype,
            'table_title': '',
            'table_rows_json': '',
            'caption_text': '',
            'visual_context_raw': '',
            'source_file': source_file,
        })
        gen_idx += 1
    
    # Tables
    for idx, tbl in enumerate(mapped_tables, start=1):
        qtype = classify_question_type(tbl.get('title', ''))
        
        rows.append({
            'gen_id': f'GEN_{gen_idx:04d}',
            'source_row_id': f'TBL_{idx:03d}',
            'use_for_generation_yn': 'Y' if tbl.get('table_rows') else 'N',
            'source_type': 'table',
            'page_no': tbl.get('page_number', ''),
            'chapter': tbl.get('chapter', ''),
            'chapter_title': tbl.get('chapter_title', ''),
            'section': tbl.get('section', ''),
            'section_title': tbl.get('section_title', ''),
            'subsection': tbl.get('subsection', ''),
            'subsection_title': tbl.get('subsection_title', ''),
            'item_title': tbl.get('title', ''),
            'source_text': tbl.get('content_text_normalized', ''),
            'source_text_raw': tbl.get('content_text_raw', ''),
            'source_text_normalized': tbl.get('content_text_normalized', ''),
            'question_type_label': qtype,
            'table_title': tbl.get('title', ''),
            'table_rows_json': tbl.get('table_rows_json', ''),
            'caption_text': '',
            'visual_context_raw': '',
            'source_file': source_file,
        })
        gen_idx += 1
    
    return pd.DataFrame(rows)


def build_support_visuals(mapped_figures: List[Dict[str, Any]], source_file: str) -> pd.DataFrame:
    """Support_Visuals 시트"""
    rows = []
    
    for idx, fig in enumerate(mapped_figures, start=1):
        qtype = classify_question_type(fig.get('caption_text', ''))
        
        rows.append({
            'row_id': f'FIG_{idx:03d}',
            'source_type': 'figure',
            'page_no': fig.get('page_number', ''),
            'chapter': fig.get('chapter', ''),
            'chapter_title': fig.get('chapter_title', ''),
            'section': fig.get('section', ''),
            'section_title': fig.get('section_title', ''),
            'subsection': fig.get('subsection', ''),
            'subsection_title': fig.get('subsection_title', ''),
            'caption_text': fig.get('caption_text', ''),
            'visual_context_raw': fig.get('context_before_raw', ''),
            'question_type_label': qtype,
            'source_file': source_file,
        })
    
    return pd.DataFrame(rows)


# =========================================================
# 14. 안내 시트
# =========================================================

def build_type_guidelines() -> pd.DataFrame:
    """Type_Guidelines"""
    rows = [
        ['정의', '용어의 의미를 직접 규정', '정의 관련 문장'],
        ['수치', '정량적 기준 (길이, 강도 등)', '수치 기준'],
        ['순서', '절차나 순서 설명', '절차 관련'],
        ['분류', '종류나 분류 설명', '분류 관련'],
        ['장점', '장점이나 효과 설명', '긍정 표현'],
        ['단점', '단점이나 문제 설명', '부정 표현'],
        ['특징', '일반적 특징 설명', '기타'],
    ]
    return pd.DataFrame(rows, columns=['question_type', 'description', 'example'])


def build_readme(pdf_path: str, output_path: str) -> pd.DataFrame:
    """README"""
    rows = [
        ['스크립트', 'extract_pdf_v8_text_only_fix.py'],
        ['입력 PDF', pdf_path],
        ['출력 Excel', output_path],
        ['시트 목록', 'Generation_Input, Type_Guidelines, Support_Visuals, README'],
        ['텍스트 추출', '완결 문장 1개 = 1 row'],
        ['문장 분리', '한국어 종결형 (소수점 무시)'],
        ['문장 복원', '토막/숫자절단 자동 병합'],
        ['검증', '저장 전 완결성 검증'],
        ['도면라벨', '완전 제거'],
        ['참조문장', '완전 제거'],
    ]
    return pd.DataFrame(rows, columns=['key', 'value'])


# =========================================================
# 15. Excel 저장
# =========================================================

def save_excel(output_path: str, df_gen: pd.DataFrame, df_type: pd.DataFrame,
               df_visuals: pd.DataFrame, df_readme: pd.DataFrame) -> None:
    """Excel 저장"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_gen.to_excel(writer, index=False, sheet_name='Generation_Input')
        df_type.to_excel(writer, index=False, sheet_name='Type_Guidelines')
        df_visuals.to_excel(writer, index=False, sheet_name='Support_Visuals')
        df_readme.to_excel(writer, index=False, sheet_name='README')
        
        for sheet in writer.sheets.values():
            for column in sheet.columns:
                max_len = 0
                col_letter = column[0].column_letter
                for cell in column:
                    try:
                        cell_len = len(str(cell.value))
                        if cell_len > max_len:
                            max_len = cell_len
                    except:
                        pass
                adjusted_width = min(max_len + 2, 50)
                sheet.column_dimensions[col_letter].width = adjusted_width
                
                for cell in column:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
    
    logger.info(f'✅ Excel 저장 완료: {output_path}')


# =========================================================
# 16. 메인
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='PDF → Excel (v8: TEXT FIX ONLY)')
    parser.add_argument('--pdf', required=True, help='입력 PDF')
    parser.add_argument('--output', default='generation_ready.xlsx', help='출력 Excel')
    args = parser.parse_args()
    
    pdf_path = args.pdf
    output_path = args.output
    
    if not os.path.exists(pdf_path):
        logger.error(f'❌ PDF 파일 없음: {pdf_path}')
        return
    
    logger.info('=' * 70)
    logger.info('📄 PDF 추출 시작 (v8: Text Complete Sentence Fix)')
    logger.info('=' * 70)
    
    # Step 1: 라인 추출
    logger.info('Step 1: 라인 추출 중...')
    lines = extract_lines_with_layout(pdf_path)
    logger.info(f'✅ {len(lines)}개 라인 추출')
    
    # Step 2: 구조 추출
    logger.info('Step 2: 문서 구조 파싱 중...')
    structure = extract_document_structure(lines, pdf_path)
    logger.info(f'✅ {len(structure["sections"])}개 절, {len(structure["subsections"])}개 소절')
    
    # Step 3: 소절 텍스트 채우기
    logger.info('Step 3: 소절 텍스트 수집 중...')
    fill_subsection_texts(structure, lines)
    logger.info('✅ 텍스트 수집 완료')
    
    # Step 4: 완결 문장 추출 (개선)
    logger.info('Step 4: 완결 문장 추출 중 (TEXT 전용)...')
    text_sentences = build_complete_text_sentences_v2(structure['subsections'])
    logger.info(f'✅ {len(text_sentences)}개 완결 문장 추출')
    
    # Step 5: 표 추출 (기존)
    logger.info('Step 5: 표 추출 중...')
    table_blocks = extract_table_blocks_from_pdf(pdf_path, lines)
    logger.info(f'✅ {len(table_blocks)}개 표 추출')
    
    # Step 6: 그림 추출 (기존)
    logger.info('Step 6: 그림 캡션 추출 중...')
    figure_blocks = extract_figure_blocks_v2(lines)
    logger.info(f'✅ {len(figure_blocks)}개 그림 추출')
    
    # Step 7: 블록 매핑
    logger.info('Step 7: 블록 매핑 중...')
    mapped_tables = map_blocks_to_subsections(structure, table_blocks)
    mapped_figures = map_blocks_to_subsections(structure, figure_blocks)
    logger.info('✅ 매핑 완료')
    
    # Step 8: Generation 행 생성
    logger.info('Step 8: Generation_Input 생성 중...')
    df_gen = build_generation_rows(text_sentences, mapped_tables, os.path.basename(pdf_path))
    logger.info(f'✅ {len(df_gen)}개 행 생성')
    
    # Step 9: Support_Visuals 생성
    logger.info('Step 9: Support_Visuals 생성 중...')
    df_visuals = build_support_visuals(mapped_figures, os.path.basename(pdf_path))
    logger.info(f'✅ {len(df_visuals)}개 그림 항목')
    
    # Step 10: 안내 시트
    logger.info('Step 10: 안내 시트 생성 중...')
    df_type = build_type_guidelines()
    df_readme = build_readme(pdf_path, output_path)
    
    # Step 11: Excel 저장
    logger.info('Step 11: Excel 저장 중...')
    save_excel(output_path, df_gen, df_type, df_visuals, df_readme)
    
    # 통계
    logger.info('=' * 70)
    logger.info('🎉 완료!')
    text_count = len(df_gen[df_gen['source_type'] == 'text'])
    table_count = len(df_gen[df_gen['source_type'] == 'table'])
    logger.info(f'📝 Complete sentences: {text_count}개')
    logger.info(f'📊 Tables: {table_count}개')
    logger.info(f'🖼️  Figures: {len(df_visuals)}개')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()