import os
import re
import nbformat as nbf

src_path = './HybridActionSpace/pepdqn_implementation.py'
dst_path = './HybridActionSpace/pepdqn_implementation.ipynb'

if not os.path.exists(src_path):
    raise FileNotFoundError(f'未找到源文件: {src_path}')

with open(src_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

nb = nbf.v4.new_notebook()
cells = []

section_code_buffer = []
section_idx = 0

def flush_code_buffer():
    if section_code_buffer:
        code_text = ''.join(section_code_buffer).rstrip()
        if code_text:
            cells.append(nbf.v4.new_code_cell(code_text))
        section_code_buffer.clear()

section_title_pattern = re.compile(r'^#\s+(\d+\.\s.*|Main Execution|Experiment Execution and Visualization)', re.IGNORECASE)
separator_pattern = re.compile(r'^#\s*={5,}')

pending_markdown = None

for i, line in enumerate(lines):
    if separator_pattern.match(line):
        # 分隔线出现 -> 刷新当前代码
        flush_code_buffer()
        continue

    m = section_title_pattern.match(line.lstrip())
    if m:
        flush_code_buffer()
        title_raw = m.group(1).strip()
        section_idx += 1
        title = title_raw
        md = f"### Section {section_idx}: {title}\n"
        cells.append(nbf.v4.new_markdown_cell(md))
        continue

    # 文件最开头的标题注释可转成简介
    if i < 10 and line.startswith('# ') and 'UAV-Assisted' in line:
        flush_code_buffer()
        intro = line.lstrip('# ').strip()
        cells.append(nbf.v4.new_markdown_cell(f"## {intro}"))
        continue

    section_code_buffer.append(line)

flush_code_buffer()

# 可添加一个最终说明 Markdown
cells.append(nbf.v4.new_markdown_cell("Notebook 自动生成完成。"))

nb['cells'] = cells

os.makedirs(os.path.dirname(dst_path), exist_ok=True)
with open(dst_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'已生成 Jupyter Notebook: {dst_path}')
