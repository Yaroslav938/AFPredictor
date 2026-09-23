#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:10:35 2026

@author: maxaugust
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Главный скрипт для запуска аннотации и/или предсказания антигрибковой активности.")
    
    # Выбор режима работы
    parser.add_argument("--mode", choices=["annotate", "predict", "all"], required=True,
                        help="Режим работы: 'annotate' (только аннотация), 'predict' (только ML предсказание), или 'all' (весь пайплайн последовательно).")

    # Аргументы для AFPredictor_annotate.py
    annotate_group = parser.add_argument_group("Аргументы для режима 'annotate' и 'all'")
    annotate_group.add_argument("--genomes", help="Путь к директории с входными геномами.")
    annotate_group.add_argument("--ext", choices=['fna', 'fasta'], help="Расширение входных файлов (fna или fasta).")
    annotate_group.add_argument("--threads", type=int, help="Количество потоков для antiSMASH.")
    annotate_group.add_argument("--out_annotate", help="Главная директория для сохранения результатов аннотации.")

    # Аргументы для AFPredictor_predict.py
    predict_group = parser.add_argument_group("Аргументы для режима 'predict' и 'all'")
    predict_group.add_argument("--jsons", help="Путь к директории с JSON файлами. В режиме 'all' указывается автоматически, если не задано вручную.")
    predict_group.add_argument("--models", help="Путь к директории с ML моделями и optimal_thresholds.json.")
    predict_group.add_argument("--out_predict", help="Директория для сохранения итоговых TSV таблиц предсказаний.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Определяем пути к скриптам, предполагая, что они лежат в той же папке
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotate_script = os.path.join(script_dir, "AFPredictor_annotate.py")
    predict_script = os.path.join(script_dir, "AFPredictor_predict.py")
    
    # Проверка наличия самих скриптов
    if args.mode in ["annotate", "all"] and not os.path.exists(annotate_script):
        sys.exit(f"Ошибка: Скрипт {annotate_script} не найден!")
    if args.mode in ["predict", "all"] and not os.path.exists(predict_script):
        sys.exit(f"Ошибка: Скрипт {predict_script} не найден!")

    # ==========================================================
    # 1. ЗАПУСК АННОТАЦИИ
    # ==========================================================
    if args.mode in ["annotate", "all"]:
        if not all([args.genomes, args.ext, args.out_annotate, args.threads]):
            sys.exit("Ошибка: Для режима 'annotate' или 'all' необходимо указать --genomes, --ext, --out_annotate и --threads.")
            
        print("\n" + "="*50)
        print(" ЗАПУСК ПАЙПЛАЙНА АННОТАЦИИ ".center(50, "="))
        print("="*50)
        
        cmd_annotate = [
            sys.executable, annotate_script,
            "-i", args.genomes,
            "-e", args.ext,
            "-o", args.out_annotate,
            "-t", str(args.threads)
        ]
        
        try:
            subprocess.run(cmd_annotate, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Процесс аннотации завершился с ошибкой: {e}")

    # ==========================================================
    # 2. ЗАПУСК ПРЕДСКАЗАНИЯ
    # ==========================================================
    if args.mode in ["predict", "all"]:
        if not all([args.models, args.out_predict]):
            sys.exit("Ошибка: Для режима 'predict' или 'all' необходимо указать --models и --out_predict.")
            
        # Логика определения папки с JSON файлами
        json_input_dir = args.jsons
        if args.mode == "all" and not json_input_dir:
            # Если это полный пайплайн, берем JSON'ы из папки final_annotation, которую создал первый скрипт
            json_input_dir = os.path.join(args.out_annotate, "final_annotation")
            
        if not json_input_dir:
            sys.exit("Ошибка: Укажите путь к директории с JSON файлами через параметр --jsons.")
            
        print("\n" + "="*50)
        print(" ЗАПУСК ПАЙПЛАЙНА ПРЕДСКАЗАНИЯ ML ".center(50, "="))
        print("="*50)

        cmd_predict = [
            sys.executable, predict_script,
            "-i", json_input_dir,
            "-m", args.models,
            "-o", args.out_predict
        ]
        
        try:
            subprocess.run(cmd_predict, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Процесс предсказания завершился с ошибкой: {e}")

if __name__ == "__main__":
    main()