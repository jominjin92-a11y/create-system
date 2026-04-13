import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =========================================================
# 1. Basic text utilities
# =========================================================

def clean_text(text: str) -> str:
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_ocr_noise(text: str) -> str:
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def trim_heading_title(title: str, max_len: int = 28) -> str:
    title = clean_text(title)
    if len(title) <= max_len:
        return title
    stop_markers = [
        '은 ', '는 ', '이 ', '가 ', '을 ', '를 ', '에서 ', '으로 ', '이며 ', '이고 ',
        '한다', '이다', '있다', '없다', '위해', '따라', '경우', '설치', '사용', '말한다'
    ]
    cut = len(title)
    for marker in stop_markers:
        idx = title.find(marker)
        if idx != -1 and idx >= 4:
            cut = min(cut, idx)
    title = title[:cut].strip(' -:·,;')
    return title[:max_len].strip()


# =========================================================
# 2. PDF line extraction with layout
# =========================================================

def _is_two_column_page(words: List[Dict[str, Any]], page_width: float) -> bool:
    if not words:
        return False
    mid_lo, mid_hi = page_width * 0.40, page_width * 0.60
    mid_words = [w for w in words if mid_lo <= (w['x0'] + w['x1']) / 2 <= mid_hi]
    ratio = len(mid_words) / len(words)
    return ratio < 0.08


def _estimate_col_split(page, words: List[Dict[str, Any]]) -> float:
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
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w['top'] / y_tol) * y_tol, w['x0']))
    lines: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_top = None

    for w in words:
        if current_top is None:
            current = [w]
            current_top = w['top']
            continue
        if abs(w['top'] - current_top) <= y_tol:
            current.append(w)
        else:
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
    lines_out: List[Dict[str, Any]] = []
    footer_margin = 60
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_height = page.height
            page_width = page.width
            words = page.extract_words() or []
            words = [w for w in words if w['bottom'] <= page_height - footer_margin]

            col_split = _estimate_col_split(page, words)
            columns = [('left', 0, page_width)] if col_split >= page_width else [('left', 0, col_split), ('right', col_split, page_width)]

            for position, x_min, x_max in columns:
                part_words = [w for w in words if w['x0'] >= x_min and w['x1'] <= x_max + 1]
                for line in _group_words_to_lines(part_words):
                    line['text'] = normalize_ocr_noise(line['text'])
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
# 3. Structure extraction
# =========================================================

def extract_chapter_meta(pdf_path: str) -> Tuple[str, str]:
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    m = re.match(r'(\d+)\.(.+)', filename)
    if m:
        return m.group(1), m.group(2)
    return '', filename


def extract_document_structure(lines: List[Dict[str, Any]], pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
    chapter_no, chapter_title = extract_chapter_meta(pdf_path)
    structure: Dict[str, List[Dict[str, Any]]] = {'sections': [], 'subsections': []}
    current_section = None
    seen_sections = set()
    seen_subsections = set()

    for line in sorted(lines, key=lambda l: (l['page_num'], 0 if l['position'] == 'left' else 1, l['top'])):
        page_number = line['page_num']
        position = line.get('position', '')
        top = line.get('top', 0.0)
        text = line['text'].strip()
        if not text:
            continue

        sub_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.+)$', text)
        if sub_match and re.search(r'[가-힣A-Za-z]', sub_match.group(2)):
            subsection = sub_match.group(1)
            parts = subsection.split('.')
            if chapter_no and parts[0] != chapter_no:
                pass
            elif subsection not in seen_subsections:
                if all(p.isdigit() and 0 < int(p) < 99 for p in parts):
                    title = trim_heading_title(sub_match.group(2))
                    row = {
                        'chapter': chapter_no,
                        'chapter_title': chapter_title,
                        'section': current_section['section'] if current_section else '',
                        'section_title': current_section['section_title'] if current_section else '',
                        'subsection': subsection,
                        'subsection_title': title,
                        'page_number': page_number,
                        'position': position,
                        'top': top,
                    }
                    structure['subsections'].append(row)
                    seen_subsections.add(subsection)
            continue

        sec_match = re.match(r'^(\d+\.\d+)\s+(.+)$', text)
        if sec_match and re.search(r'[가-힣A-Za-z]', sec_match.group(2)):
            section = sec_match.group(1)
            parts = section.split('.')
            if chapter_no and parts[0] != chapter_no:
                pass
            elif section not in seen_sections:
                if all(p.isdigit() and 0 < int(p) < 99 for p in parts):
                    title = trim_heading_title(sec_match.group(2))
                    current_section = {
                        'chapter': chapter_no,
                        'chapter_title': chapter_title,
                        'section': section,
                        'section_title': title,
                        'page_number': page_number,
                        'position': position,
                        'top': top,
                    }
                    structure['sections'].append(current_section)
                    seen_sections.add(section)
    return structure


def fill_subsection_texts(structure: Dict[str, List[Dict[str, Any]]], lines: List[Dict[str, Any]]) -> None:
    def line_sort_key(l):
        return (l['page_num'], 0 if l.get('position', 'left') == 'left' else 1, l.get('top', 0))

    def sub_sort_key(s):
        return (s['page_number'], 0 if s.get('position', 'left') == 'left' else 1, s.get('top', 0))

    subsections = sorted(structure['subsections'], key=sub_sort_key)
    sorted_lines = sorted(lines, key=line_sort_key)

    for sub in subsections:
        sub['text'] = ''

    for idx, sub in enumerate(subsections):
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
        sub['text'] = clean_text('\n'.join(texts))


# =========================================================
# 4. Table extraction
# =========================================================

def _line_matches_table_caption(text: str) -> bool:
    return bool(re.match(r'^\[표\s*\d+-\d+\]', text.strip()))


def _line_matches_figure_caption(text: str) -> bool:
    return bool(re.match(r'^\[그림\s*\d+-\d+\]', text.strip()))


def _position_from_x(x_center: float, col_split: float, page_width: float) -> str:
    if col_split >= page_width:
        return 'left'
    return 'left' if x_center < col_split else 'right'


def _is_junk_table(rows: List[List]) -> bool:
    """
    [FIX 표-오탐] 목차 점선, 페이지 장식 등 쓰레기 표 필터링
    - 점선(…·) 비율이 높거나 실질 텍스트가 거의 없는 표 제거
    """
    if not rows:
        return True
    all_text = ' '.join(str(c) for row in rows for c in row if c)
    if not all_text.strip():
        return True
    dot_chars = sum(1 for c in all_text if c in '…·.·⋯')
    if len(all_text) > 0 and dot_chars / len(all_text) > 0.3:
        return True
    # CHAPTER, 헤더성 반복 텍스트
    if 'CHAPTER' in all_text and len(all_text) < 200:
        return True
    # 셀이 1개뿐인 표 (분리선 오탐)
    total_cells = sum(1 for row in rows for c in row if c and str(c).strip())
    if total_cells <= 2:
        return True
    return False


def _merge_none_cells(rows: List[List]) -> List[List]:
    """
    [FIX 표-2] 병합 셀(None) 처리 — 중복 행 방지
    왼쪽→위쪽 순으로 채우되, 완전 동일한 연속 행은 제거
    """
    if not rows:
        return rows
    filled = [list(row) for row in rows]
    num_cols = max(len(r) for r in filled)

    for row in filled:
        while len(row) < num_cols:
            row.append('')

    # 왼쪽으로 채우기 (colspan)
    for r_idx, row in enumerate(filled):
        for c_idx in range(1, len(row)):
            if row[c_idx] is None:
                row[c_idx] = row[c_idx - 1] if row[c_idx - 1] is not None else ''

    # 위쪽으로 채우기 (rowspan)
    for c_idx in range(num_cols):
        for r_idx in range(1, len(filled)):
            if c_idx < len(filled[r_idx]) and (filled[r_idx][c_idx] is None or filled[r_idx][c_idx] == ''):
                if c_idx < len(filled[r_idx - 1]):
                    filled[r_idx][c_idx] = filled[r_idx - 1][c_idx]

    # [FIX] 완전히 동일한 연속 행 제거 (중복 방지)
    deduped = []
    for row in filled:
        if not deduped or row != deduped[-1]:
            deduped.append(row)
    return deduped


def extract_table_blocks_from_pdf(pdf_path: str, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    caption_lines = [l for l in lines if _line_matches_table_caption(l['text'])]
    caption_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for cap in caption_lines:
        caption_by_page[cap['page_num']].append(cap)

    tables_out: List[Dict[str, Any]] = []
    used_caption_keys = set()

    col_splits: Dict[int, float] = {}
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

                # [FIX 표-오탐] 쓰레기 표 필터링
                if _is_junk_table(raw_rows):
                    logger.info('[표 오탐 필터] p%d tbl%d 제거 (점선/장식/빈표)', page_idx, tbl_idx)
                    continue

                cleaned_rows = []
                for row_no, row in enumerate(raw_rows, start=1):
                    row = [clean_text(str(c)) if c is not None else '' for c in row]
                    if any(cell for cell in row):
                        cleaned_rows.append({'row_no': row_no, 'cells': row, 'raw': ' | '.join([c for c in row if c])})
                if not cleaned_rows:
                    continue

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

                tables_out.append({
                    'title': title,
                    'type': 'table',
                    'page_number': page_idx,
                    'position': position,
                    'top': top,
                    'bottom': bottom,
                    'bbox': tbl.bbox,
                    'content_lines': [r['raw'] for r in cleaned_rows],
                    'content_text_raw': '\n'.join(r['raw'] for r in cleaned_rows),
                    'content_text_normalized': clean_text('\n'.join(r['raw'] for r in cleaned_rows)),
                    'table_rows': cleaned_rows,
                    'content_json': json.dumps(cleaned_rows, ensure_ascii=False),
                    'label': '표',
                })

    for cap in caption_lines:
        key = (cap['page_num'], cap['text'])
        if key in used_caption_keys:
            continue
        logger.warning('[고아 캡션] 표 자동탐지 실패: page=%d, caption="%s"', cap['page_num'], cap['text'])
        tables_out.append({
            'title': cap['text'],
            'type': 'table',
            'page_number': cap['page_num'],
            'position': cap.get('position', ''),
            'top': cap.get('top', 0),
            'bottom': cap.get('bottom', 0),
            'bbox': None,
            'content_lines': [],
            'content_text_raw': '',
            'content_text_normalized': '',
            'table_rows': [],
            'content_json': '[]',
            'label': '표',
            'note': 'caption only; table not auto-detected',
        })

    return sorted(tables_out, key=lambda x: (x['page_number'], x.get('position', ''), x.get('top', 0)))


# =========================================================
# 5. Figure extraction
# =========================================================

_SOURCE_PATTERN = re.compile(r'출처\s*[:：]|source\s*[:：]', re.IGNORECASE)


def _clean_figure_caption(text: str) -> str:
    """
    [FIX 그림-1] 캡션 앞에 붙은 본문 조각 제거
    - '[그림' 이전 텍스트 제거
    - 캡션 내부의 '출처 :' 이후 URL 제거
    """
    text = clean_text(text)
    # '[그림' 앞에 붙은 본문 텍스트 제거
    m = re.search(r'\[그림\s*\d+-\d+\]', text)
    if m:
        text = text[m.start():]
    # 캡션 내 '출처 :' 이후 제거 [FIX 그림-2]
    text = re.sub(r'\s*출처\s*[:：]\s*\S+', '', text).strip()
    text = re.sub(r'\s*[Ss]ource\s*[:：]\s*\S+', '', text).strip()
    return text


def _split_figure_caption_line(text: str) -> List[str]:
    text = clean_text(text)
    matches = list(re.finditer(r'\[그림\s*\d+-\d+\]', text))
    if not matches:
        return []
    if len(matches) == 1:
        return [_clean_figure_caption(text)]
    parts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part = _clean_figure_caption(text[start:end].strip())
        if part:
            parts.append(part)
    return parts


def _get_image_bboxes_by_page(pdf_path: str) -> Dict[int, List[Dict[str, Any]]]:
    result: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for img in (page.images or []):
                result[page_num].append({
                    'x0': img.get('x0', 0),
                    'x1': img.get('x1', 0),
                    'top': img.get('top', 0),
                    'bottom': img.get('bottom', 0),
                })
    return result


def extract_figure_blocks_from_lines(lines: List[Dict[str, Any]], pdf_path: str = '') -> List[Dict[str, Any]]:
    figures: List[Dict[str, Any]] = []

    image_bboxes: Dict[int, List[Dict[str, Any]]] = {}
    if pdf_path:
        try:
            image_bboxes = _get_image_bboxes_by_page(pdf_path)
        except Exception as e:
            logger.warning('이미지 bbox 추출 실패: %s', e)

    lines_by_col: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
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

            # context 수집: 출처/다른캡션/헤딩 제외
            prev_context = []
            j = idx - 1
            while j >= 0 and len(prev_context) < 6:
                prev_txt = col_lines[j]['text'].strip()
                if (prev_txt
                        and not _line_matches_figure_caption(prev_txt)
                        and not _line_matches_table_caption(prev_txt)
                        and not _SOURCE_PATTERN.search(prev_txt)
                        and not re.match(r'^(\d+\.\d+\.\d+|\d+\.\d+)\s+', prev_txt)):
                    prev_context.append(prev_txt)
                j -= 1
            prev_context.reverse()

            page_num = line['page_num']
            cap_top = line.get('top', 0)
            cap_pos = line.get('position', '')

            img_top = cap_top
            page_imgs = image_bboxes.get(page_num, [])
            best_img = None
            best_dist = float('inf')
            for img in page_imgs:
                dist = cap_top - img.get('bottom', 0)
                if 0 <= dist < best_dist:
                    best_dist = dist
                    best_img = img
            if best_img:
                img_top = best_img.get('top', cap_top)

            for part in caption_parts:
                figures.append({
                    'title': part,
                    'type': 'figure',
                    'page_number': page_num,
                    'position': cap_pos,
                    'top': img_top,
                    'bottom': cap_top,
                    'content_lines': [],
                    'content_text_raw': '',
                    'content_text_normalized': '',
                    'context_before_raw': '\n'.join(prev_context),
                    'content_json': '',
                    'label': '그림',
                })
    return figures


# =========================================================
# 6. Block mapping to subsections
# =========================================================

def map_blocks_to_subsections(structure: Dict[str, List[Dict[str, Any]]], blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            'content_lines': info.get('content_lines', []),
            'content_text_raw': info.get('content_text_raw', ''),
            'content_text_normalized': info.get('content_text_normalized', ''),
            'table_rows': info.get('table_rows', []),
            'content_json': info.get('content_json', ''),
            'context_before_raw': info.get('context_before_raw', ''),
        })
    return results


# =========================================================
# 7. Generation helpers
# =========================================================

def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    raw_parts = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[①-⑳⑴-⑽•·\-]\s*', line) or re.match(r'^\(?\d+\)\s*', line):
            raw_parts.append(line)
            continue
        parts = re.split(r'(?<=[다요음])\.\s+|(?<=다)\s+(?=[①-⑳⑴-⑽])|(?<=다)\s+(?=\d+\))', line)
        raw_parts.extend([p.strip() for p in parts if p.strip()])
    return raw_parts


def classify_question_type(text: str) -> str:
    t = text.strip()
    if re.search(r'(을 말한다|를 말한다|의미한다|정의)', t):
        return '정의'
    if re.search(r'\d', t) and re.search(r'(m|mm|cm|kgf|kW|포대|%|이상|이하|이내|초과|미만|최고|최소|폭|높이|간격|강도)', t):
        return '수치'
    if re.search(r'(순서|먼저|이후|다음의 그림과 같은 순서|후에|역순)', t):
        return '순서'
    if re.search(r'(분류할 수 있다|종류|구분|구성되어|로 나뉜다|로 구성)', t):
        return '분류'
    if re.search(r'(장점|유리|절감|향상|효과적|바람직하다)', t):
        return '장점'
    if re.search(r'(단점|문제점|불리|주의가 필요|영향을 미치지 않는|제한)', t):
        return '단점'
    return '특징'


def askable(text: str, source_type: str) -> Tuple[str, float]:
    t = text.strip()
    if source_type == 'figure':
        if not re.search(r'\[그림\s*\d+-\d+\]', t):
            return 'N', 0.1   # 캡션 아닌 행은 제외
        return 'Y', 0.55
    if len(t) < 15:
        return 'N', 0.15
    if re.match(r'^\[그림', t):
        return 'N', 0.1
    score = 0.75
    qtype = classify_question_type(t)
    if qtype in ['정의', '수치', '순서', '분류']:
        score = 0.92
    elif qtype in ['장점', '단점', '특징']:
        score = 0.82
    return 'Y', score


def stem_template(qtype: str) -> str:
    mapping = {
        '정의': '다음 설명에 해당하는 것으로 옳은 것은?',
        '수치': '다음 중 기준으로 옳은 것은?',
        '순서': '다음 중 순서로 옳은 것은?',
        '분류': '다음 중 분류로 옳은 것은?',
        '장점': '다음 중 장점으로 옳은 것은?',
        '단점': '다음 중 단점으로 옳지 않은 것은?',
        '특징': '다음 중 설명으로 옳은 것은?',
    }
    return mapping.get(qtype, '다음 중 옳은 것은?')


def answer_option_type(qtype: str) -> str:
    mapping = {
        '정의': '용어형', '수치': '수치형', '순서': '절차형',
        '분류': '종류형', '장점': '설명형', '단점': '설명형', '특징': '설명형',
    }
    return mapping.get(qtype, '설명형')


def polarity_allowed(qtype: str) -> Tuple[str, str]:
    if qtype in ['정의', '수치', '순서', '분류']:
        return 'Y', 'N'
    return 'Y', 'Y'


# =========================================================
# 8. Build generation sheets
# =========================================================

def _fmt_chapter(val) -> str:
    """[FIX 타입] chapter/section float → str 변환"""
    if val is None or val == '':
        return ''
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)


def _fmt_section(val) -> str:
    if val is None or val == '':
        return ''
    s = str(val)
    # '2.3' 형태 유지, '2.30' → '2.3'
    try:
        parts = s.split('.')
        return '.'.join(str(int(p)) for p in parts)
    except Exception:
        return s


def build_generation_rows(
    structure: Dict[str, List[Dict[str, Any]]],
    mapped_tables: List[Dict[str, Any]],
    mapped_figures: List[Dict[str, Any]],
    source_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    visuals = []
    gen_idx = 1

    for sub in structure['subsections']:
        base = {
            'page_no': sub.get('page_number', ''),
            'chapter': _fmt_chapter(sub.get('chapter', '')),
            'chapter_title': sub.get('chapter_title', ''),
            'section': _fmt_section(sub.get('section', '')),
            'section_title': sub.get('section_title', ''),
            'subsection': sub.get('subsection', ''),
            'subsection_title': sub.get('subsection_title', ''),
            'source_file': source_file,
        }
        for sent_idx, sent in enumerate(split_sentences(sub.get('text', '')), start=1):
            qtype = classify_question_type(sent)
            ask_yn, ask_score = askable(sent, 'text')
            pos, neg = polarity_allowed(qtype)
            rows.append({
                'gen_id': f'GEN_{gen_idx:04d}',
                'source_row_id': f"TXT_{sub.get('subsection','').replace('.','_')}_{sent_idx:03d}",
                'use_for_generation_yn': ask_yn,
                'source_type': 'text',
                **base,
                'item_title': sub.get('subsection_title', ''),
                'source_text': sent,
                'source_text_raw': sent,
                'source_text_normalized': clean_text(sent),
                'question_type_label': qtype,
                'askable_score': ask_score,
                'stem_template': stem_template(qtype),
                'answer_option_type': answer_option_type(qtype),
                'positive_allowed_yn': pos,
                'negative_allowed_yn': neg,
                'distractor_group_id': f"DG_{sub.get('section','').replace('.','_')}_{sub.get('subsection','').replace('.','_')}_{qtype}",
                'distractor_scope': 'same_subsection',
                'distractor_rule': 'same subsection / same type / semantically similar but not identical',
                'generation_priority': 'high' if qtype in ['정의', '수치', '분류', '순서'] else 'medium',
                'linked_support_visuals': '',
                'source_tracking_key': f"{sub.get('subsection','')}|p{sub.get('page_number','')}|{sub.get('position','')}|text|{sent_idx}",
                'table_title': '',
                'table_rows_json': '',
                'visual_context_raw': '',
                'note': '',
            })
            gen_idx += 1

    for idx, blk in enumerate(mapped_tables, start=1):
        qtype = classify_question_type(blk.get('content_text_normalized') or blk.get('title', ''))
        ask_yn, ask_score = askable(blk.get('content_text_normalized') or blk.get('title', ''), 'table')
        pos, neg = polarity_allowed(qtype)
        common = {
            'page_no': blk.get('page_number', ''),
            'chapter': _fmt_chapter(blk.get('chapter', '')),
            'chapter_title': blk.get('chapter_title', ''),
            'section': _fmt_section(blk.get('section', '')),
            'section_title': blk.get('section_title', ''),
            'subsection': blk.get('subsection', ''),
            'subsection_title': blk.get('subsection_title', ''),
            'source_file': source_file,
        }
        rows.append({
            'gen_id': f'GEN_{gen_idx:04d}',
            'source_row_id': f'TBL_{idx:03d}',
            'use_for_generation_yn': 'Y' if blk.get('table_rows') else 'N',
            'source_type': 'table',
            **common,
            'item_title': blk.get('title', ''),
            'source_text': blk.get('content_text_normalized', ''),
            'source_text_raw': blk.get('content_text_raw', ''),
            'source_text_normalized': blk.get('content_text_normalized', ''),
            'question_type_label': qtype,
            'askable_score': ask_score,
            'stem_template': stem_template(qtype),
            'answer_option_type': answer_option_type(qtype),
            'positive_allowed_yn': pos,
            'negative_allowed_yn': neg,
            'distractor_group_id': f"DG_{blk.get('section','').replace('.','_')}_{blk.get('subsection','').replace('.','_')}_{qtype}",
            'distractor_scope': 'same_section',
            'distractor_rule': 'same section / same table family / preserve row-column relation',
            'generation_priority': 'high',
            'linked_support_visuals': '',
            'source_tracking_key': f"{blk.get('subsection','')}|p{blk.get('page_number','')}|{blk.get('position','')}|table|{idx}",
            'table_title': blk.get('title', ''),
            'table_rows_json': json.dumps(blk.get('table_rows', []), ensure_ascii=False),
            'visual_context_raw': '',
            'note': blk.get('note', 'table raw preserved'),
        })
        gen_idx += 1

    for idx, blk in enumerate(mapped_figures, start=1):
        qtype = classify_question_type(blk.get('context_before_raw') or blk.get('title', ''))
        ask_yn, ask_score = askable(blk.get('title', ''), 'figure')
        pos, neg = polarity_allowed(qtype)
        common = {
            'page_no': blk.get('page_number', ''),
            'chapter': _fmt_chapter(blk.get('chapter', '')),
            'chapter_title': blk.get('chapter_title', ''),
            'section': _fmt_section(blk.get('section', '')),
            'section_title': blk.get('section_title', ''),
            'subsection': blk.get('subsection', ''),
            'subsection_title': blk.get('subsection_title', ''),
            'source_file': source_file,
        }
        rows.append({
            'gen_id': f'GEN_{gen_idx:04d}',
            'source_row_id': f'FIG_{idx:03d}',
            'use_for_generation_yn': ask_yn,
            'source_type': 'figure',
            **common,
            'item_title': blk.get('title', ''),
            'source_text': blk.get('title', ''),
            'source_text_raw': blk.get('title', ''),
            'source_text_normalized': clean_text(blk.get('title', '')),
            'question_type_label': qtype,
            'askable_score': ask_score,
            'stem_template': stem_template(qtype),
            'answer_option_type': answer_option_type(qtype),
            'positive_allowed_yn': pos,
            'negative_allowed_yn': neg,
            'distractor_group_id': f"DG_{blk.get('section','').replace('.','_')}_{blk.get('subsection','').replace('.','_')}_{qtype}",
            'distractor_scope': 'same_subsection',
            'distractor_rule': 'figure caption + context / same subsection',
            'generation_priority': 'low',
            'linked_support_visuals': f'FIG_{idx:03d}',
            'source_tracking_key': f"{blk.get('subsection','')}|p{blk.get('page_number','')}|{blk.get('position','')}|figure|{idx}",
            'table_title': '',
            'table_rows_json': '',
            'visual_context_raw': blk.get('context_before_raw', ''),
            'note': 'figure caption; refer to Support_Visuals for context',
        })
        gen_idx += 1

        visuals.append({
            'row_id': f'FIG_{idx:03d}',
            'source_type': 'figure',
            'page_no': blk.get('page_number', ''),
            'chapter': _fmt_chapter(blk.get('chapter', '')),
            'chapter_title': blk.get('chapter_title', ''),
            'section': _fmt_section(blk.get('section', '')),
            'section_title': blk.get('section_title', ''),
            'subsection': blk.get('subsection', ''),
            'subsection_title': blk.get('subsection_title', ''),
            'item_title': blk.get('title', ''),
            'visual_content_summary': blk.get('title', ''),
            'caption_text': blk.get('title', ''),
            'visual_context_raw': blk.get('context_before_raw', ''),
            'linked_paragraph_raw': blk.get('context_before_raw', ''),
            'question_type_label': qtype,
            'visual_use_level': 'support',
            'recommended_use': 'caption + nearby paragraph in same column as support evidence',
            'note': '',
            'source_file': source_file,
        })

    return pd.DataFrame(rows), pd.DataFrame(visuals)


def build_distractor_pools(df_gen: pd.DataFrame) -> pd.DataFrame:
    if df_gen.empty:
        return pd.DataFrame()
    pools = []
    for dg, grp in df_gen[df_gen['use_for_generation_yn'] == 'Y'].groupby('distractor_group_id'):
        pools.append({
            'distractor_group_id': dg,
            'section': grp.iloc[0]['section'],
            'subsection': grp.iloc[0]['subsection'],
            'question_type_label': grp.iloc[0]['question_type_label'],
            'candidate_count': len(grp),
            'candidate_row_ids': ', '.join(grp['source_row_id'].astype(str).tolist()),
            'candidate_titles': ' || '.join(grp['item_title'].fillna('').tolist()),
            'candidate_texts': ' || '.join(grp['source_text_raw'].fillna('').tolist()),
        })
    return pd.DataFrame(pools)


def build_type_guidelines() -> pd.DataFrame:
    rows = [
        ['정의', '용어의 의미를 직접 규정하는 문장', '다음 설명에 해당하는 것으로 옳은 것은?', '용어형', 'Y', 'N', 'same subsection', '동일 범위 유사 개념 사용'],
        ['수치', '길이 높이 폭 강도 등 정량 기준 문장', '다음 중 기준으로 옳은 것은?', '수치형', 'Y', 'N', 'same subsection', '같은 단위/근접 수치 사용'],
        ['순서', '절차나 시공순서를 설명하는 문장', '다음 중 순서로 옳은 것은?', '절차형', 'Y', 'N', 'same subsection', '단계 순서 변형'],
        ['분류', '종류나 체계를 나누는 문장', '다음 중 분류로 옳은 것은?', '종류형', 'Y', 'N', 'same section', '같은 범주의 다른 종류 사용'],
        ['장점', '유리한 점/효과 설명', '다음 중 장점으로 옳은 것은?', '설명형', 'Y', 'Y', 'same subsection', '유사 개념과 혼동되는 설명 사용'],
        ['단점', '문제점/불리한 점 설명', '다음 중 옳지 않은 것은?', '설명형', 'Y', 'Y', 'same subsection', '긍정 설명과 혼합 금지'],
        ['특징', '목적 기능 구성 설치사항 일반 속성 설명', '다음 중 설명으로 옳은 것은?', '설명형', 'Y', 'Y', 'same subsection', '같은 소절 내 유사 설명 사용'],
    ]
    return pd.DataFrame(rows, columns=['question_type_label', 'definition', 'stem_template', 'answer_option_type', 'positive_allowed_yn', 'negative_allowed_yn', 'distractor_scope', 'distractor_rule'])


def build_readme(pdf_path: str, output_path: str) -> pd.DataFrame:
    rows = [
        ['script', 'build_generation_excel_split_layers_v3.py'],
        ['input_pdf', pdf_path],
        ['output_excel', output_path],
        ['sheets', 'Generation_Input, Distractor_Pools, Type_Guidelines, Support_Visuals, README'],
        ['fix_v2_버그1', 'fill_subsection_texts: 전역 정렬 기반 다음 페이지 텍스트 수집'],
        ['fix_v2_버그2', 'extract_document_structure: lines 직접 사용으로 top 정보 보존'],
        ['fix_v2_버그3', '소절/섹션 번호 챕터 범위 검증 추가'],
        ['fix_v2_버그4', 'map_blocks_to_subsections: position 숫자 변환으로 튜플 비교 오류 수정'],
        ['fix_v2_버그5', '_estimate_col_split: 2단 여부 먼저 판별 후 split 적용'],
        ['fix_v2_표1', '캡션-표 position 통일: _position_from_x 함수로 기준 일원화'],
        ['fix_v2_표2', '병합 셀(None) 처리: _merge_none_cells로 colspan/rowspan 모사'],
        ['fix_v2_표3', '표 탐지 전략 3단계 fallback (lines→lines+text→text+text)'],
        ['fix_v2_그림1', 'Generation_Input에 그림 행 추가 (source_type=figure)'],
        ['fix_v2_그림2', 'page.images로 실제 이미지 bbox 추출해 position/top 보정'],
        ['fix_v2_그림3', 'context_before_raw에서 출처/연속 그림 캡션 필터링'],
        ['fix_v3_그림A', '_clean_figure_caption: [그림 앞 본문 조각 제거 + 캡션 내 출처URL 제거'],
        ['fix_v3_그림B', 'askable: 캡션 아닌 figure 행은 use_for_generation_yn=N 처리'],
        ['fix_v3_표A', '_is_junk_table: 목차 점선/CHAPTER 헤더/빈표 오탐 필터링'],
        ['fix_v3_표B', '_merge_none_cells: 완전 동일 연속 행 중복 제거'],
        ['fix_v3_타입', '_fmt_chapter/_fmt_section: chapter/section float→str 정규화'],
    ]
    return pd.DataFrame(rows, columns=['key', 'value'])


def save_excel(output_path: str, df_gen: pd.DataFrame, df_pools: pd.DataFrame,
               df_type: pd.DataFrame, df_visuals: pd.DataFrame, df_readme: pd.DataFrame) -> None:
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_gen.to_excel(writer, index=False, sheet_name='Generation_Input')
        df_pools.to_excel(writer, index=False, sheet_name='Distractor_Pools')
        df_type.to_excel(writer, index=False, sheet_name='Type_Guidelines')
        df_visuals.to_excel(writer, index=False, sheet_name='Support_Visuals')
        df_readme.to_excel(writer, index=False, sheet_name='README')
    logger.info('Saved: %s', output_path)


# =========================================================
# 9. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='PDF -> generation-ready Excel (v3)')
    parser.add_argument('--pdf', required=True, help='Input PDF path')
    parser.add_argument('--output', default='generation_ready.xlsx', help='Output xlsx path')
    args = parser.parse_args()

    pdf_path = args.pdf
    output_path = args.output

    logger.info('Step 1: PDF 라인 추출')
    lines = extract_lines_with_layout(pdf_path)

    logger.info('Step 2: 문서 구조 추출')
    structure = extract_document_structure(lines, pdf_path)
    logger.info('  sections: %d, subsections: %d', len(structure['sections']), len(structure['subsections']))

    logger.info('Step 3: 소절 텍스트 채우기')
    fill_subsection_texts(structure, lines)

    logger.info('Step 4: 표 추출')
    table_blocks = extract_table_blocks_from_pdf(pdf_path, lines)
    logger.info('  tables found: %d', len(table_blocks))

    logger.info('Step 5: 그림 추출')
    figure_blocks = extract_figure_blocks_from_lines(lines, pdf_path=pdf_path)
    logger.info('  figures found: %d', len(figure_blocks))

    logger.info('Step 6: 블록 → 소절 매핑')
    mapped_tables = map_blocks_to_subsections(structure, table_blocks)
    mapped_figures = map_blocks_to_subsections(structure, figure_blocks)

    logger.info('Step 7: Generation 행 생성')
    df_gen, df_visuals = build_generation_rows(structure, mapped_tables, mapped_figures, os.path.basename(pdf_path))
    df_pools = build_distractor_pools(df_gen)
    df_type = build_type_guidelines()
    df_readme = build_readme(pdf_path, output_path)

    logger.info('Step 8: Excel 저장')
    save_excel(output_path, df_gen, df_pools, df_type, df_visuals, df_readme)
    logger.info('완료: Generation_Input 행 수 = %d', len(df_gen))


if __name__ == '__main__':
    main()
