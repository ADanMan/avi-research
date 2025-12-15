#!/usr/bin/env python3
"""
Интерактивный запуск эксперимента с переключением VPN

Этапы:
1. Прогон запросов (Baseline + AVI) - требует ВНУТРЕННИЙ VPN
2. LLM Judge оценка - требует ВНЕШНИЙ VPN
3. Полный прогон (всё сразу - не рекомендуется)

Usage:
    python scripts/03_run_experiment.py
"""

import sys
import asyncio
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment.runner import ExperimentRunner
from src.utils.helpers import ExperimentConfig


def print_menu():
    print("=" * 60)
    print("🚀 AVI Experiment Runner")
    print("=" * 60)
    print()
    print("Выберите этап:")
    print()
    print("1. Прогон запросов (Baseline + AVI)")
    print("   → Требует: ВНУТРЕННИЙ VPN для Cotype и AVI")
    print("   → Результат: queries_results.csv (без Judge оценок)")
    print()
    print("2. LLM Judge оценка")
    print("   → Требует: ВНЕШНИЙ VPN для OpenAI")
    print("   → Вход: queries_results.csv")
    print("   → Результат: final_results.csv (с Judge оценками)")
    print()
    print("3. Полный прогон (запросы + judge)")
    print("   → Не рекомендуется (нужно переключать VPN)")
    print()
    print("0. Выход")
    print()


async def run_queries_only(config, test_queries):
    """Этап 1: Только запросы к моделям (внутренний VPN)"""
    print("=" * 60)
    print("📡 ЭТАП 1: Прогон запросов")
    print("=" * 60)
    print()
    print("⚠️  ПРОВЕРЬТЕ: Включен ВНУТРЕННИЙ VPN для Cotype и AVI")
    input("   Нажмите Enter когда готовы...")
    print()

    runner = ExperimentRunner(
        avi_api_url=config.get_with_env('avi.api_url', 'AVI_API_URL', 'http://localhost:8000'),
        avi_api_key=config.get_with_env('avi.api_key', 'AVI_API_KEY'),
        test_model=config.get_with_env('llm.test_model', 'COTYPE_MODEL', 'cotype-2.5-pro'),
        test_api_base=config.get_with_env('llm.test_api_base', 'COTYPE_API_BASE'),
        test_api_key=config.get_with_env('llm.test_api_key', 'COTYPE_API_KEY'),
    )

    results = await runner.run_queries_only(
        test_queries_df=test_queries,
        show_progress=True,
    )

    # Сохранить промежуточные результаты
    output_path = Path("data/results/queries_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print()
    print(f"✅ Сохранено: {output_path}")
    print(f"📊 Обработано запросов: {len(results)}")
    print()
    print("Следующий шаг:")
    print("  1. Переключитесь на ВНЕШНИЙ VPN")
    print("  2. Запустите этап 2 (LLM Judge)")
    print()

    return results


async def run_judge_only(config):
    """Этап 2: Только LLM Judge оценка (внешний VPN)"""
    print("=" * 60)
    print("⚖️  ЭТАП 2: LLM Judge Оценка")
    print("=" * 60)
    print()

    # Проверить наличие queries_results.csv
    queries_path = Path("data/results/queries_results.csv")
    if not queries_path.exists():
        print(f"❌ Ошибка: {queries_path} не найден!")
        print("   Сначала запустите этап 1 (Прогон запросов)")
        return None

    print("⚠️  ПРОВЕРЬТЕ: Включен ВНЕШНИЙ VPN для OpenAI")
    input("   Нажмите Enter когда готовы...")
    print()

    # Загрузить результаты запросов
    queries_df = pd.read_csv(queries_path)
    print(f"📂 Загружено запросов: {len(queries_df)}")
    print()

    # Запустить Judge
    from src.experiment.llm_judge import LLMJudge
    from tqdm import tqdm

    judge = LLMJudge(
        model=config.get_with_env('llm.judge_model', 'OPENAI_MODEL', 'gpt-4o-mini')
    )

    print("🔍 Оценка ответов с LLM Judge...")
    judgments = []

    for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Judge evaluation"):
        judgment = judge.evaluate(
            query=row['query'],
            expected_answer=row['expected_answer'],
            policy=row.get('policy', ''),
            system_response=row['avi_response'],
        )
        judgments.append(judgment)

    # Добавить оценки к результатам
    queries_df['llm_compliance_score'] = [j['compliance_score'] for j in judgments]
    queries_df['llm_helpfulness_score'] = [j['helpfulness_score'] for j in judgments]
    queries_df['llm_naturalness_score'] = [j['naturalness_score'] for j in judgments]
    queries_df['llm_reasoning'] = [j['reasoning'] for j in judgments]
    queries_df['llm_detected_issues'] = [str(j['detected_issues']) for j in judgments]

    # Сохранить финальные результаты
    final_path = Path("data/results/final_results.csv")
    queries_df.to_csv(final_path, index=False)

    print()
    print(f"✅ Сохранено: {final_path}")
    print()
    print("📊 Итоговая статистика:")
    print(f"   Compliance rate: {(queries_df['llm_compliance_score'] >= 0.5).mean():.1%}")
    print(f"   Mean helpfulness: {queries_df['llm_helpfulness_score'].mean():.2f}/1.0")
    print()

    return queries_df


async def run_full_experiment(config, test_queries):
    """Этап 3: Полный прогон (не рекомендуется)"""
    print("⚠️  Полный прогон требует переключения VPN во время работы!")
    print("   Рекомендуется запускать этапы 1 и 2 отдельно.")
    confirm = input("   Продолжить? (y/N): ")

    if confirm.lower() != 'y':
        return None

    # Сначала запросы
    await run_queries_only(config, test_queries)

    print()
    print("=" * 60)
    print("⚠️  ПЕРЕКЛЮЧИТЕ VPN!")
    print("=" * 60)
    print("Переключитесь с ВНУТРЕННЕГО на ВНЕШНИЙ VPN для OpenAI")
    input("Нажмите Enter когда переключили...")

    # Потом Judge
    await run_judge_only(config)


async def main():
    config = ExperimentConfig("config/experiment_config.yaml")

    while True:
        print_menu()
        choice = input("Выберите этап (0-3): ").strip()

        if choice == "0":
            print("Выход.")
            return 0

        if choice in ["1", "2", "3"]:
            # Загрузить test queries если нужно
            test_queries = None
            if choice in ["1", "3"]:
                test_queries_path = Path("data/processed/test_queries.csv")
                if not test_queries_path.exists():
                    print(f"❌ Ошибка: {test_queries_path} не найден!")
                    print(f"   Сначала запустите: python scripts/02_transform_dataset.py")
                    input("\nНажмите Enter для продолжения...")
                    continue

                test_queries = pd.read_csv(test_queries_path)

            # Запустить выбранный этап
            if choice == "1":
                await run_queries_only(config, test_queries)
            elif choice == "2":
                await run_judge_only(config)
            elif choice == "3":
                await run_full_experiment(config, test_queries)

            input("\nНажмите Enter для продолжения...")
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
