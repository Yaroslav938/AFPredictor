#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:10:35 2026

@author: maxaugust
"""

import os
import json
import glob
import subprocess
import shutil
from datetime import datetime
from Bio import SeqIO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Автоматизация пайплайна antiSMASH: черновая аннотация, извлечение кластеров и финальная аннотация.")
    
    parser.add_argument("-i", "--input", required=True, 
                        help="Путь к директории с входными геномами.")
    parser.add_argument("-e", "--ext", required=True, choices=['fna', 'fasta'], 
                        help="Расширение входных файлов (fna или fasta).")
    parser.add_argument("-o", "--output", required=True, 
                        help="Главная директория для сохранения результатов (например, out_antismash).")
    parser.add_argument("-t", "--threads", required=True, type=int, 
                        help="Количество потоков.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Переносим переменные из argparse для удобства использования ниже
    input_dir = args.input
    ext = args.ext
    out_dir = args.output
    threads = str(args.threads) # subprocess требует, чтобы аргументы были строками
    
    draft_dir = os.path.join(out_dir, "draft_annotation")
    clusters_dir = os.path.join(out_dir, "clusters")
    final_dir = os.path.join(out_dir, "final_annotation")

    os.makedirs(draft_dir, exist_ok=True)
    os.makedirs(clusters_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    print(f"Директории для пайплайна успешно созданы в: {out_dir}")

    # Списки для проверки гибридов
    hybride_single = [
        'NRPS', 'NRPS-like', 'NRP-metallophore', 'isocyanide-nrp', 
        'thioamide-NRP', 'T1PKS', 'T2PKS', 'T3PKS', 'HR-T2PKS', 
        'transAT-PKS', 'transAT-PKS-like', 'PKS-like', 'PpyS-KS', 
        'hglE-KS', 'betalactone','chemical_hybrid' , 'interleaved', 'arylpolyene',
        'RRE-containing'
    ]

    input_files = glob.glob(os.path.join(input_dir, f"*.{ext}"))

    if not input_files:
        print(f"Файлы с расширением {ext} не найдены в папке {input_dir}")
        return # Вместо exit(1) используем return для безопасного выхода из функции

    # =====================================================================
    # ЭТАП 1: ЧЕРНОВАЯ АННОТАЦИЯ И ПОДКЛАСТЕРИЗАЦИЯ
    # =====================================================================
    print("\n=== ЧЕРНОВАЯ АННОТАЦИЯ И ИЗВЛЕЧЕНИЕ СУБКЛАСТЕРОВ ===")

    for file_path in input_files:
        base = os.path.basename(file_path).replace(f".{ext}", "")
        base_out_dir = os.path.join(draft_dir, base) 
        
        print("=================================================")
        print(f"\033[1m{base}\033[0m START TIME \033[1m{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\033[0m")
        
        # Файлы-маркеры успешного завершения
        json_file = os.path.join(base_out_dir, f"{base}.json")
        gbk_file = os.path.join(base_out_dir, f"{base}.gbk")
        
        # 1. ПРОВЕРКА НА ПРОПУСК
        if os.path.exists(json_file) and os.path.exists(gbk_file):
            print(f"  [ПРОПУСК] Черновая аннотация для {base} уже существует.")
        else:
            # Если папка существует, но маркеров нет (прервано посередине)
            if os.path.exists(base_out_dir):
                print(f"  [ОЧИСТКА] Удаление незавершенной папки {base_out_dir}...")
                shutil.rmtree(base_out_dir)
                
            cmd_rough = [
                "antismash", file_path, "-c", threads,
                "--output-dir", base_out_dir,
                "--output-basename", base,
                "--genefinding-tool", "prodigal",
                "--logfile", os.path.join(base_out_dir, f"{base}_log"),
                "--no-zip-output", "--no-region-gbks", "--allow-long-headers", "--enable-html"
            ]
            
            subprocess.run(cmd_rough, check=True)
        
        # 2. Извлечение подкластеров в формате GBK
        if os.path.exists(json_file) and os.path.exists(gbk_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            gbk_records = {rec.id: rec for rec in SeqIO.parse(gbk_file, "genbank")}
                
            regions = []
            region_counter = 1 
            
            for v1 in data.get('records', []):
                contig_id = v1.get('id', 'contig') 
                
                if contig_id not in gbk_records:
                    print(f" [Предупреждение] Контиг {contig_id} не найден в {gbk_file}")
                    continue
                    
                parent_record = gbk_records[contig_id]
                
                for area in v1.get('areas', []):
                    current_region_candidates = []
                    
                    for candidate in area.get('candidates', []):
                        original_kind = candidate.get('kind', 'unknown') 
                        
                        if original_kind == 'neighbouring':
                            continue
                        
                        if original_kind == 'single':
                            proto_index = str(candidate['protoclusters'][0]) 
                            proto_data = area['protoclusters'][proto_index]
                            
                            start = proto_data['start']
                            end = proto_data['end']
                            core_start = proto_data['core_start']
                            core_end = proto_data['core_end']
                            kind = proto_data['product'] 
                        else:
                            start = candidate['start']
                            end = candidate['end']
                            core_start = start + 20000
                            core_end = end - 20000
                            kind = original_kind 
                            
                        current_region_candidates.append({
                            'start': start, 'end': end,
                            'core_start': core_start, 'core_end': core_end,
                            'kind': kind
                        })
                        
                    current_region_candidates.sort(key=lambda x: x['end'] - x['start'])
                    filtered_candidates = []
                    
                    for i, cand in enumerate(current_region_candidates):
                        keep_candidate = True
                        if cand['kind'] in hybride_single:
                            for larger_cand in current_region_candidates[i+1:]:
                                if cand['core_start'] >= larger_cand['start'] and cand['core_end'] <= larger_cand['end']:
                                    keep_candidate = False
                                    break 
                        
                        if keep_candidate:
                            filtered_candidates.append(cand)
                            
                    filtered_candidates.sort(key=lambda x: x['start'])
                    final_candidate_counter = 1
                    
                    for cand in filtered_candidates:
                        start = cand['start']
                        end = cand['end']
                        
                        seq_name = f"{base}_{contig_id}_BGC_{region_counter}_subcluster_{final_candidate_counter}"
                        
                        sub_record = parent_record[start:end]
                        sub_record.id = seq_name
                        sub_record.name = seq_name[:16] 
                        sub_record.description = f"Extracted BGC subcluster from {contig_id} ({start}-{end})"
                        
                        regions.append((seq_name, sub_record))
                        final_candidate_counter += 1
                        
                    region_counter += 1

            # 3. Сохранение извлеченных подкластеров в папку clusters
            for seq_name, sub_record in regions:
                out_gbk_path = os.path.join(clusters_dir, f"{seq_name}.gbk")
                SeqIO.write(sub_record, out_gbk_path, "genbank")
                    
            print(f"Извлечено подкластеров (GBK): {len(regions)}")
        else:
            print(f"Внимание: JSON файл {json_file} или GBK файл {gbk_file} не найден!")

        print(f"\033[1m{base}\033[0m END TIME \033[1m{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\033[0m")
        print("=================================================\n")


    # =====================================================================
    # ЭТАП 2: ФИНАЛЬНАЯ АННОТАЦИЯ СУБКЛАСТЕРОВ
    # =====================================================================
    print("\n=== ФИНАЛЬНАЯ АННОТАЦИЯ ===")

    cluster_files = glob.glob(os.path.join(clusters_dir, "*.gbk"))

    if not cluster_files:
        print(f"Субкластеры для финальной аннотации не найдены в {clusters_dir}")
        return

    for file_path in cluster_files:
        base = os.path.basename(file_path).replace(".gbk", "")
        final_out_dir = os.path.join(final_dir, base)
        
        print("=================================================")
        print(f"\033[1m{base}\033[0m START TIME \033[1m{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\033[0m")
        
        # Файл-маркер успешного завершения финальной аннотации
        final_json = os.path.join(final_out_dir, f"{base}.json")
        
        # ПРОВЕРКА НА ПРОПУСК
        if os.path.exists(final_json):
            print(f"  [ПРОПУСК] Финальная аннотация для {base} уже существует.")
            print(f"\033[1m{base}\033[0m END TIME \033[1m{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\033[0m")
            print("=================================================\n")
            continue # Переходим сразу к следующему файлу в цикле
            
        # Если папка существует, но маркеров нет (прервано посередине)
        if os.path.exists(final_out_dir):
            print(f"  [ОЧИСТКА] Удаление незавершенной папки {final_out_dir}...")
            shutil.rmtree(final_out_dir)
        
        cmd_final = [
            "antismash", file_path, "-c", threads,
            "--output-dir", final_out_dir,
            "--output-basename", base,
            "--logfile", os.path.join(final_out_dir, f"{base}_log"), 
            "--cc-mibig", "--clusterhmmer", "--cb-knownclusters", "--tfbs",
            "--no-zip-output", "--no-region-gbks", "--allow-long-headers", "--enable-html"
        ]
        
        subprocess.run(cmd_final, check=True)
        
        print(f"\033[1m{base}\033[0m END TIME \033[1m{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\033[0m")
        print("=================================================\n")

    print("Все задачи успешно завершены!")

# Точка входа скрипта теперь находится в самом конце
if __name__ == "__main__":
    main()