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
# 1. 텍스트 정제 유틸리티
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
# 2. PDF 라인 추출 (레이아웃 보존)
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
    """
    PDF에서 레이아웃 정보 포함해 라인 추출
    반환: [{text, page_num, position(left/right), top, x0, x1, bottom}, ...]
    """
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
# 3. 문서 구조 추출 (장/절/소절)
# =========================================================

def extract_document_structure(lines: List[Dict[str, Any]], pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    문서 구조 추출 (장/절/소절)
    """
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
        
        # 소절 패턴: 2.3.1 제목
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
        
        # 절 패턴: 2.3 제목
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
    """
    각 소절에 원문 텍스트 채우기 (최대한 보존)
    """
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
            
            # 현재 소절 번호 라인 제외
            if re.match(rf'^{re.escape(sub["subsection"])}\s+', txt):
                continue
            
            # 다음 절/소절 번호 라인 제외
            if re.match(r'^\d+\.\d+\.\d+\s+', txt) or re.match(r'^\d+\.\d+\s+', txt):
                continue
            
            texts.append(txt)
        
        sub['text'] = '\n'.join(texts)


# =========================================================
# 4. 표 추출 (Raw 구조 유지)
# =========================================================

def _is_junk_table(rows: List[List]) -> bool:
    """쓰레기 표 필터링 (목차 점선, 페이지 장식 등)"""
    if not rows:
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
    """병합 셀(None) 처리"""
    if not rows:
        return rows
    
    filled = [list(row) for row in rows]
    num_cols = max(len(r) for r in filled) if filled else 0
    
    for row in filled:
        while len(row) < num_cols:
            row.append('')
    
    # 왼쪽으로 채우기
    for r_idx, row in enumerate(filled):
        for c_idx in range(1, len(row)):
            if row[c_idx] is None:
                row[c_idx] = row[c_idx - 1] if row[c_idx - 1] is not None else ''
    
    # 위쪽으로 채우기
    for c_idx in range(num_cols):
        for r_idx in range(1, len(filled)):
            if c_idx < len(filled[r_idx]) and (filled[r_idx][c_idx] is None or filled[r_idx][c_idx] == ''):
                if c_idx < len(filled[r_idx - 1]):
                    filled[r_idx][c_idx] = filled[r_idx - 1][c_idx]
    
    # 중복 제거
    deduped = []
    for row in filled:
        if not deduped or row != deduped[-1]:
            deduped.append(row)
    
    return deduped


def _position_from_x(x_center: float, col_split: float, page_width: float) -> str:
    """좌표에서 position 결정"""
    if col_split >= page_width:
        return 'left'
    return 'left' if x_center < col_split else 'right'


def _line_matches_table_caption(text: str) -> bool:
    """표 캡션 패턴 매칭"""
    return bool(re.match(r'^\[표\s*\d+-\d+\]', text.strip()))


def extract_table_blocks_from_pdf(pdf_path: str, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    표 추출 (raw 구조 유지)
    반환: [{title, type, page_number, position, top, table_rows, content_text_raw, ...}, ...]
    """
    caption_lines = [l for l in lines if _line_matches_table_caption(l['text'])]
    caption_by_page = defaultdict(list)
    for cap in caption_lines:
        caption_by_page[cap['page_num']].append(cap)
    
    tables_out = []
    used_caption_keys = set()
    
    col_splits = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words() or []
            col_splits[page_idx] = _estimate_col_split(page, words)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_caps = caption_by_page.get(page_idx, [])
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
                
                # 쓰레기 표 필터링
                if _is_junk_table(raw_rows):
                    logger.info(f'[표 오탐 필터] p{page_idx} tbl{tbl_idx} 제거')
                    continue
                
                cleaned_rows = []
                for row in raw_rows:
                    row = [clean_text(str(c)) if c is not None else '' for c in row]
                    if any(cell for cell in row):
                        cleaned_rows.append(row)
                
                if not cleaned_rows:
                    continue
                
                # 캡션 매칭
                chosen_cap = None
                best_score = None
                for cap in page_caps:
                    cap_center_x = (cap.get('x0', 0) + cap.get('x1', cap.get('x0', 0))) / 2
                    cap_position = _position_from_x(cap_center_x, col_split, page.width)
                    if cap_position != position:
                        continue
                    cap_top = cap.get('top', 0)
                    distance = min(abs(cap_top - top), abs(cap_top - bottom))
                    score = distance + (0 if cap_top <= top + 30 else 80)
                    if best_score is None or score < best_score:
                        chosen_cap = cap
                        best_score = score
                
                title = chosen_cap['text'] if chosen_cap else f'[표 {page_idx}-{tbl_idx}]'
                if chosen_cap:
                    used_caption_keys.add((chosen_cap['page_num'], chosen_cap['text']))
                
                # 원문 텍스트 생성
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
    
    return sorted(tables_out, key=lambda x: (x['page_number'], x.get('position', ''), x.get('top', 0)))


# =========================================================
# 5. 그림 캡션 추출 (실제 캡션만)
# =========================================================

def _clean_figure_caption(text: str) -> str:
    """그림 캡션 정제 (참조문장 제거)"""
    text = clean_text(text)
    
    # '[그림' 앞의 본문 제거
    m = re.search(r'\[그림\s*\d+-\d+\]', text)
    if m:
        text = text[m.start():]
    
    # 캡션 내 '출처' 이후 제거
    text = re.sub(r'\s*출처\s*[:：]\s*\S+', '', text).strip()
    text = re.sub(r'\s*[Ss]ource\s*[:：]\s*\S+', '', text).strip()
    
    # "[표/그림 x-y]와 같이", "[표/그림 x-y]에 제시된" 같은 참조 제외
    if re.search(r'\[(?:표|그림|Figure|Table)\s+\d+-\d+\](?:와 같이|에 제시된|와 함께|을 참고)', text):
        return ''  # 이런 텍스트는 캡션이 아님
    
    return text


def _split_figure_caption_line(text: str) -> List[str]:
    """한 라인에 여러 그림 캡션이 있으면 분리"""
    text = clean_text(text)
    matches = list(re.finditer(r'\[그림\s*\d+-\d+\]', text))
    if not matches:
        return []
    if len(matches) == 1:
        cleaned = _clean_figure_caption(text)
        return [cleaned] if cleaned else []
    
    parts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part = _clean_figure_caption(text[start:end].strip())
        if part:
            parts.append(part)
    
    return parts


def extract_figure_blocks_from_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    그림 캡션 추출 (실제 캡션만)
    반환: [{title, type, page_number, position, top, context_before_raw, ...}, ...]
    """
    figures = []
    
    lines_by_col = defaultdict(list)
    for line in lines:
        lines_by_col[(line['page_num'], line.get('position', ''))].append(line)
    
    for key, col_lines in lines_by_col.items():
        col_lines = sorted(col_lines, key=lambda x: x.get('top', 0))
        
        for idx, line in enumerate(col_lines):
            txt = line['text']
            if '[그림' not in txt:
                continue
            
            caption_parts = _split_figure_caption_line(txt)
            if not caption_parts:
                continue
            
            # 앞의 문맥 수집 (캡션 아닌 부분, 출처 제외)
            prev_context = []
            j = idx - 1
            while j >= 0 and len(prev_context) < 6:
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
            
            for part in caption_parts:
                figures.append({
                    'title': part,
                    'type': 'figure',
                    'page_number': page_num,
                    'position': cap_pos,
                    'top': cap_top,
                    'caption_text': part,
                    'context_before_raw': '\n'.join(prev_context),
                })
    
    return figures


# =========================================================
# 6. 블록을 소절에 매핑
# =========================================================

def map_blocks_to_subsections(structure: Dict[str, List[Dict[str, Any]]], blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    표/그림을 소절에 매핑 (page + position + top 기준)
    """
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
        
        results.append({
            'chapter': matched.get('chapter', '') if matched else '',
            'chapter_title': matched.get('chapter_title', '') if matched else '',
            'section': matched.get('section', '') if matched else '',
            'section_title': matched.get('section_title', '') if matched else '',
            'subsection': matched.get('subsection', '') if matched else '',
            'subsection_title': matched.get('subsection_title', '') if matched else '',
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
# 7. Generation 행 생성
# =========================================================

def classify_question_type(text: str) -> str:
    """문제 유형 분류"""
    t = text.strip()
    if re.search(r'(을 말한다|를 말한다|의미한다|정의)', t):
        return '정의'
    if re.search(r'\d', t) and re.search(r'(m|mm|cm|kgf|kW|%|이상|이하|강도|높이)', t):
        return '수치'
    if re.search(r'(순서|먼저|이후|다음의|후에|역순)', t):
        return '순서'
    if re.search(r'(분류|종류|구분|구성|나뉜)', t):
        return '분류'
    if re.search(r'(장점|유리|절감|향상|효과)', t):
        return '장점'
    if re.search(r'(단점|문제|불리|주의)', t):
        return '단점'
    return '특징'


def build_generation_rows(structure: Dict[str, List[Dict[str, Any]]], 
                          mapped_tables: List[Dict[str, Any]],
                          mapped_figures: List[Dict[str, Any]],
                          source_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generation_Input 및 Support_Visuals 시트 생성
    """
    gen_rows = []
    visual_rows = []
    gen_idx = 1
    
    # 1. 텍스트 (본문)
    for sub in structure['subsections']:
        text = sub.get('text', '').strip()
        if not text:
            continue
        
        qtype = classify_question_type(text)
        
        gen_rows.append({
            'gen_id': f'GEN_{gen_idx:04d}',
            'source_row_id': f"TXT_{sub.get('subsection', '').replace('.', '_')}",
            'use_for_generation_yn': 'Y',
            'source_type': 'text',
            'page_no': sub.get('page_number', ''),
            'chapter': sub.get('chapter', ''),
            'chapter_title': sub.get('chapter_title', ''),
            'section': sub.get('section', ''),
            'section_title': sub.get('section_title', ''),
            'subsection': sub.get('subsection', ''),
            'subsection_title': sub.get('subsection_title', ''),
            'item_title': sub.get('subsection_title', ''),
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
    
    # 2. 표
    for idx, tbl in enumerate(mapped_tables, start=1):
        qtype = classify_question_type(tbl.get('title', ''))
        
        gen_rows.append({
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
    
    # 3. 그림
    for idx, fig in enumerate(mapped_figures, start=1):
        qtype = classify_question_type(fig.get('caption_text', ''))
        
        gen_rows.append({
            'gen_id': f'GEN_{gen_idx:04d}',
            'source_row_id': f'FIG_{idx:03d}',
            'use_for_generation_yn': 'Y',
            'source_type': 'figure',
            'page_no': fig.get('page_number', ''),
            'chapter': fig.get('chapter', ''),
            'chapter_title': fig.get('chapter_title', ''),
            'section': fig.get('section', ''),
            'section_title': fig.get('section_title', ''),
            'subsection': fig.get('subsection', ''),
            'subsection_title': fig.get('subsection_title', ''),
            'item_title': fig.get('title', ''),
            'source_text': fig.get('caption_text', ''),
            'source_text_raw': fig.get('caption_text', ''),
            'source_text_normalized': normalize_text(fig.get('caption_text', '')),
            'question_type_label': qtype,
            'table_title': '',
            'table_rows_json': '',
            'caption_text': fig.get('caption_text', ''),
            'visual_context_raw': fig.get('context_before_raw', ''),
            'source_file': source_file,
        })
        
        visual_rows.append({
            'row_id': f'FIG_{idx:03d}',
            'source_type': 'figure',
            'page_no': fig.get('page_number', ''),
            'chapter': fig.get('chapter', ''),
            'section': fig.get('section', ''),
            'subsection': fig.get('subsection', ''),
            'subsection_title': fig.get('subsection_title', ''),
            'caption_text': fig.get('caption_text', ''),
            'visual_context_raw': fig.get('context_before_raw', ''),
            'source_file': source_file,
        })
        
        gen_idx += 1
    
    return pd.DataFrame(gen_rows), pd.DataFrame(visual_rows)


def build_type_guidelines() -> pd.DataFrame:
    """Type_Guidelines 시트"""
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
    """README 시트"""
    rows = [
        ['스크립트', 'extract_pdf_v3.py'],
        ['입력 PDF', pdf_path],
        ['출력 Excel', output_path],
        ['시트 목록', 'Generation_Input, Type_Guidelines, Support_Visuals, README'],
        ['추출 전략', '본문/표/그림 분리 파이프라인'],
        ['표 처리', 'Raw 구조 유지, table_rows_json 저장'],
        ['그림 처리', '실제 캡션만 추출, 참조 문장 제외'],
        ['소절 매핑', 'page + position + top 기준'],
    ]
    return pd.DataFrame(rows, columns=['key', 'value'])


def save_excel(output_path: str, df_gen: pd.DataFrame, df_type: pd.DataFrame,
               df_visuals: pd.DataFrame, df_readme: pd.DataFrame) -> None:
    """Excel 저장"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_gen.to_excel(writer, index=False, sheet_name='Generation_Input')
        df_type.to_excel(writer, index=False, sheet_name='Type_Guidelines')
        df_visuals.to_excel(writer, index=False, sheet_name='Support_Visuals')
        df_readme.to_excel(writer, index=False, sheet_name='README')
        
        # 자동 열 너비 조정
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
# 8. 메인
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='PDF → Excel (본문/표/그림 분리)')
    parser.add_argument('--pdf', required=True, help='입력 PDF')
    parser.add_argument('--output', default='generation_ready.xlsx', help='출력 Excel')
    args = parser.parse_args()
    
    pdf_path = args.pdf
    output_path = args.output
    
    if not os.path.exists(pdf_path):
        logger.error(f'❌ PDF 파일 없음: {pdf_path}')
        return
    
    logger.info('=' * 70)
    logger.info('📄 PDF 추출 시작 (본문/표/그림 분리 파이프라인)')
    logger.info('=' * 70)
    
    # Step 1: 라�� 추출
    logger.info('Step 1: 레이아웃 정보 포함해 라인 추출 중...')
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
    
    # Step 4: 표 추출
    logger.info('Step 4: 표 추출 중...')
    table_blocks = extract_table_blocks_from_pdf(pdf_path, lines)
    logger.info(f'✅ {len(table_blocks)}개 표 추출')
    
    # Step 5: 그림 추출
    logger.info('Step 5: 그림 캡션 추출 중...')
    figure_blocks = extract_figure_blocks_from_lines(lines)
    logger.info(f'✅ {len(figure_blocks)}개 그림 캡션 추출')
    
    # Step 6: 블록 매핑
    logger.info('Step 6: 블록을 소절에 매핑 중...')
    mapped_tables = map_blocks_to_subsections(structure, table_blocks)
    mapped_figures = map_blocks_to_subsections(structure, figure_blocks)
    logger.info(f'✅ 매핑 완료')
    
    # Step 7: Generation 행 생성
    logger.info('Step 7: Generation 행 생성 중...')
    df_gen, df_visuals = build_generation_rows(structure, mapped_tables, mapped_figures, 
                                                os.path.basename(pdf_path))
    logger.info(f'✅ {len(df_gen)}개 행 생성')
    
    # Step 8: 안내 시트 생성
    logger.info('Step 8: 안내 시트 생성 중...')
    df_type = build_type_guidelines()
    df_readme = build_readme(pdf_path, output_path)
    
    # Step 9: Excel 저장
    logger.info('Step 9: Excel 저장 중...')
    save_excel(output_path, df_gen, df_type, df_visuals, df_readme)
    
    logger.info('=' * 70)
    logger.info('🎉 완료!')
    # ✅ 수정: DataFrame 접근 방식 변경
    text_count = len(df_gen[df_gen['source_type'] == 'text'])
    table_count = len(df_gen[df_gen['source_type'] == 'table'])
    figure_count = len(df_gen[df_gen['source_type'] == 'figure'])
    logger.info(f'📊 본문: {text_count}개')
    logger.info(f'📊 표: {table_count}개')
    logger.info(f'📊 그림: {figure_count}개')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()