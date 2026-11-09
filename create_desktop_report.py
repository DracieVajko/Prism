from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

wb = Workbook()

# Sheet 1: Model Comparison
ws1 = wb.active
ws1.title = 'Model Comparison'

headers = ['Model', 'Size', 'Device', 'Language', 'Confidence', 'Segments', 'Punctuation', 'Status', 'Notes']
data = [
    ['Whisper Turbo', '~1.6 GB', 'CPU', 'en', '1.00', '12', 'Yes', 'Working', 'Recommended default for transcription'],
    ['Whisper Large', '~3.0 GB', 'CPU', 'en', '0.99', '12', 'Partial', 'Working', 'Larger, slower than turbo'],
    ['Nemotron ASR 0.6B', '~2.5 GB', 'CUDA', 'auto', '—', '1', 'Yes', 'Working', 'GPU acceleration, some errors'],
    ['Gemma4 12B Q4_K_M GGUF', '~7.1 GB', 'CPU', '—', '—', '—', 'Yes', 'Working (post-processing)', 'Summarization only, via user LLM loader'],
    ['Gemma4 12B BF16', '~24 GB', 'CPU', '—', '—', '—', '—', 'Failed (OOM)', 'Not enough system RAM'],
    ['Gemma4 E2B-2B', '~10 GB', 'CPU', '—', '—', '—', 'Artifacts', 'Failed', 'Corrupted output (pints... artifacts)'],
]

ws1.append(headers)
for row in data:
    ws1.append(row)

for col in range(1, len(headers)+1):
    cell = ws1.cell(row=1, column=col)
    cell.font = Font(bold=True, color='FFFFFF')
    cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    cell.alignment = Alignment(horizontal='center')

for col in range(1, len(headers)+1):
    max_len = max(len(str(ws1.cell(row=r, column=col).value)) for r in range(1, len(data)+2))
    ws1.column_dimensions[get_column_letter(col)].width = min(max_len + 2, 50)

# Sheet 2: File Locations
ws2 = wb.create_sheet('File Locations')
file_headers = ['File Type', 'Path']
file_data = [
    ['Original Video', r'D:\Projekty\AI Income Challenge\day1\2026-11-07 20-39-31.mp4'],
    ['Test Audio (WAV)', r'D:\AI\Prism\testdata\video\test_1min_audio.wav'],
    ['Test Video Clip', r'D:\AI\Prism\testdata\video\test_1min.mp4'],
    ['Comparison Outputs', r'D:\AI\Prism\testdata\transcription_comparison_v2'],
    ['PC Server', r'D:\AI\Prism\pc\unified_server.py'],
    ['LLM Providers Module', r'D:\AI\Prism\pc\unified\llm_providers.py'],
    ['Android API Server', r'D:\AI\Prism\android\app\src\main\java\eu\odysseus\diffusion\ApiServer.kt'],
    ['Android Build', r'D:\AI\Prism\android\app\build.gradle.kts'],
    ['PC Models Directory', r'D:\AI\Prism\pc\models'],
    ['Gemma4 12B Q4_K_M GGUF', r'D:\AI\Prism\pc\models\gemma-4-12B-it-q4\gemma-4-12b-it-Q4_K_M.gguf'],
    ['Desktop Report (MD)', r'C:\Users\DracieVajko\Desktop\PRISM_LLM_Model_Comparison.md'],
    ['Desktop Report (XLSX)', r'C:\Users\DracieVajko\Desktop\PRISM_LLM_Model_Comparison.xlsx'],
]

ws2.append(file_headers)
for row in file_data:
    ws2.append(row)

for col in range(1, len(file_headers)+1):
    cell = ws2.cell(row=1, column=col)
    cell.font = Font(bold=True, color='FFFFFF')
    cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    cell.alignment = Alignment(horizontal='center')

for col in range(1, len(file_headers)+1):
    max_len = max(len(str(ws2.cell(row=r, column=col).value)) for r in range(1, len(file_data)+2))
    ws2.column_dimensions[get_column_letter(col)].width = min(max_len + 2, 80)

# Sheet 3: Configuration
ws3 = wb.create_sheet('Configuration')
config_headers = ['Environment Variable', 'Default', 'Description']
config_data = [
    ['PRISM_LLM_PROVIDER', 'none', 'ollama | lmstudio | openai | anthropic | custom | none'],
    ['PRISM_LLM_BASE_URL', '', 'http://localhost:11434 (Ollama), http://localhost:1234 (LM Studio)'],
    ['PRISM_LLM_API_KEY', '', 'API key for public providers'],
    ['PRISM_LLM_MODEL', 'gemma4:12b', 'Model name (recommended default in dropdown)'],
    ['PRISM_LLM_ENDPOINT', '', 'Custom endpoint path'],
    ['PRISM_PROMPT_SUMMARY', 'Summarize the text into concise notes with key points.', 'Default prompt for summary operation'],
    ['PRISM_PROMPT_CLEANUP', 'Fix grammar, spelling, and punctuation. Keep meaning unchanged.', 'Default prompt for cleanup operation'],
    ['PRISM_PROMPT_GRAMMAR', 'Fix grammar and spelling only. Do not rewrite or summarize.', 'Default prompt for grammar operation'],
    ['PRISM_PROMPT_LIST', 'Extract a bulleted list of the key items from the text.', 'Default prompt for list operation'],
    ['PRISM_PROMPT_TRANSLATE', 'Translate the text into {target_lang}. Reply with translation only.', 'Default prompt for translate operation'],
    ['PRISM_PROMPT_REWRITE', 'Rewrite the text to be clearer and easier to read.', 'Default prompt for rewrite operation'],
    ['PRISM_PROMPT_MEETING_NOTES', 'Summarize meeting transcript with date, participants, topic, decisions, action items.', 'Default prompt for meeting_notes operation'],
]

ws3.append(config_headers)
for row in config_data:
    ws3.append(row)

for col in range(1, len(config_headers)+1):
    cell = ws3.cell(row=1, column=col)
    cell.font = Font(bold=True, color='FFFFFF')
    cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    cell.alignment = Alignment(horizontal='center')

for col in range(1, len(config_headers)+1):
    max_len = max(len(str(ws3.cell(row=r, column=col).value)) for r in range(1, len(config_data)+2))
    ws3.column_dimensions[get_column_letter(col)].width = min(max_len + 2, 60)

# Sheet 4: Default Prompts
ws4 = wb.create_sheet('Default Prompts')
prompt_headers = ['Operation', 'Default Prompt', 'Env Override']
prompt_data = [
    ['summary', 'Summarize the text into concise notes with key points.', 'PRISM_PROMPT_SUMMARY'],
    ['cleanup', 'Fix grammar, spelling, and punctuation. Keep meaning unchanged.', 'PRISM_PROMPT_CLEANUP'],
    ['grammar', 'Fix grammar and spelling only. Do not rewrite or summarize.', 'PRISM_PROMPT_GRAMMAR'],
    ['list', 'Extract a bulleted list of the key items from the text.', 'PRISM_PROMPT_LIST'],
    ['translate', 'Translate the text into {target_lang}. Reply with translation only.', 'PRISM_PROMPT_TRANSLATE'],
    ['rewrite', 'Rewrite the text to be clearer and easier to read.', 'PRISM_PROMPT_REWRITE'],
    ['meeting_notes', 'Summarize meeting transcript with date, participants, topic, decisions, action items.', 'PRISM_PROMPT_MEETING_NOTES'],
]

ws4.append(prompt_headers)
for row in prompt_data:
    ws4.append(row)

for col in range(1, len(prompt_headers)+1):
    cell = ws4.cell(row=1, column=col)
    cell.font = Font(bold=True, color='FFFFFF')
    cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    cell.alignment = Alignment(horizontal='center')

for col in range(1, len(prompt_headers)+1):
    max_len = max(len(str(ws4.cell(row=r, column=col).value)) for r in range(1, len(prompt_data)+2))
    ws4.column_dimensions[get_column_letter(col)].width = min(max_len + 2, 70)

wb.save(r'C:\Users\DracieVajko\Desktop\PRISM_LLM_Model_Comparison.xlsx')
print('Excel saved successfully')
