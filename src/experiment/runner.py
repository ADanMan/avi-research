"""
Experiment Runner

Runs complete experiment: baseline + AVI, collects all metrics
"""

import os
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import httpx
from .evaluator import AutomaticEvaluator
from .llm_judge import LLMJudge
from ..utils.helpers import create_experiment_dir, save_json


class ExperimentRunner:
    """Run full experiment on AVI system"""

    def __init__(
        self,
        avi_api_url: Optional[str] = None,
        avi_api_key: Optional[str] = None,
        test_model: Optional[str] = None,
        test_api_base: Optional[str] = None,
        test_api_key: Optional[str] = None,
    ):
        """
        Initialize experiment runner

        Args:
            avi_api_url: AVI API URL (e.g., http://localhost:8000)
            avi_api_key: AVI API key
            test_model: Model to test (e.g., cotype-2.5-pro)
            test_api_base: API base for test model
            test_api_key: API key for test model
        """
        # Get values from params or environment variables
        self.avi_url = (avi_api_url or os.getenv("AVI_API_URL", "http://localhost:8000")).rstrip('/')
        self.avi_api_key = avi_api_key or os.getenv("AVI_API_KEY")
        
        if not self.avi_api_key:
            raise ValueError("AVI_API_KEY must be provided or set in environment")

        self.test_model = test_model or os.getenv("COTYPE_MODEL", "cotype-2.5-pro")
        self.test_api_base = test_api_base or os.getenv("COTYPE_API_BASE")
        self.test_api_key = test_api_key or os.getenv("COTYPE_API_KEY")

        self.evaluator = AutomaticEvaluator()
        self.judge = LLMJudge()

    async def run_full_experiment(
        self,
        test_queries_df: pd.DataFrame,
        run_baseline: bool = True,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run complete experiment

        Args:
            test_queries_df: Test queries DataFrame
            run_baseline: Whether to run baseline (unfiltered)
            output_dir: Where to save results (auto-creates if None)
            show_progress: Show progress bars

        Returns:
            Results DataFrame with all metrics

        Example:
            >>> runner = ExperimentRunner(
            ...     avi_api_url="http://localhost:8000",
            ...     avi_api_key="your-key"
            ... )
            >>> results = await runner.run_full_experiment(test_queries_df)
        """
        # Create experiment directory
        if output_dir is None:
            exp_dir = create_experiment_dir()
        else:
            exp_dir = Path(output_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Experiment directory: {exp_dir}")
        print(f"📊 Running {len(test_queries_df)} queries...")
        print()

        results = []

        iterator = test_queries_df.iterrows()
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Running experiment")

        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx, row in iterator:
                # 1. Run baseline (if requested)
                baseline_response = None
                baseline_latency = None

                if run_baseline:
                    baseline_start = time.time()
                    baseline_response = await self._query_baseline(
                        client, row['query']
                    )
                    baseline_latency = (time.time() - baseline_start) * 1000

                # 2. Run AVI
                avi_start = time.time()
                avi_response = await self._query_avi(
                    client, row['query']
                )
                avi_latency = (time.time() - avi_start) * 1000

                # 3. Automatic evaluation
                auto_metrics = self.evaluator.evaluate_single_query(
                    query=row['query'],
                    expected_answer=row['expected_answer'],
                    system_response=avi_response['response'],
                    filter_result=avi_response.get('input_filter_result', {}),
                    latency_ms=avi_latency,
                )

                # 4. LLM Judge evaluation
                llm_judgment = self.judge.evaluate(
                    query=row['query'],
                    expected_answer=row['expected_answer'],
                    policy=row.get('policy', ''),
                    system_response=avi_response['response'],
                )

                # 5. Combine results
                result = {
                    'query_id': row.get('id', f"query_{idx}"),
                    'company': row.get('company', ''),

                    # Baseline
                    'baseline_response': baseline_response,
                    'baseline_latency_ms': baseline_latency,

                    # AVI
                    'avi_response': avi_response['response'],
                    'avi_latency_ms': avi_latency,
                    'latency_overhead_ms': avi_latency - (baseline_latency or 0),

                    # Automatic metrics
                    **auto_metrics,

                    # LLM Judge
                    'llm_compliance_score': llm_judgment['compliance_score'],
                    'llm_helpfulness_score': llm_judgment['helpfulness_score'],
                    'llm_naturalness_score': llm_judgment['naturalness_score'],
                    'llm_reasoning': llm_judgment['reasoning'],
                    'llm_detected_issues': str(llm_judgment['detected_issues']),
                }

                results.append(result)

                # Save checkpoint every 10 queries
                if (idx + 1) % 10 == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_df.to_csv(
                        exp_dir / f"checkpoint_{idx+1}.csv",
                        index=False
                    )

        # Create final DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv(exp_dir / "raw_results.csv", index=False)
        print(f"\n💾 Saved results to {exp_dir}/raw_results.csv")

        # Compute aggregate metrics
        agg_metrics = self.evaluator.aggregate_results(results)
        save_json(agg_metrics, exp_dir / "aggregate_metrics.json")
        print(f"💾 Saved aggregate metrics to {exp_dir}/aggregate_metrics.json")

        return results_df

    async def _query_baseline(
        self,
        client: httpx.AsyncClient,
        query: str,
    ) -> str:
        """Query baseline (unfiltered) model - direct Cotype without AVI"""
        
        # Используем LLMClient для запроса к Cotype
        from ..utils.llm_client import LLMClient
        
        baseline_llm = LLMClient(
            provider="cotype",
            model=self.test_model,
            api_key=self.test_api_key,
            api_base=self.test_api_base,
            temperature=0.7,
            max_tokens=2000,
        )
        
        # Простой запрос без фильтрации
        response = baseline_llm.generate(query)
        return response

    async def _query_avi(
        self,
        client: httpx.AsyncClient,
        query: str,
    ) -> Dict[str, Any]:
        """Query AVI system"""
        url = f"{self.avi_url}/api/v1/query"

        headers = {
            "X-API-Key": self.avi_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "query": query,
            "rag_mode": True,
            "stream": False,
        }

        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()
    
    async def run_queries_only(
        self,
        test_queries_df: pd.DataFrame,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Прогон только запросов (без LLM Judge)
        Используется когда нужен другой VPN для Judge
        """
        print(f"📊 Прогон {len(test_queries_df)} запросов...")
        print()

        results = []
        iterator = test_queries_df.iterrows()
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Queries")

        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx, row in iterator:
                # Baseline
                baseline_start = time.time()
                baseline_response = await self._query_baseline(client, row['query'])
                baseline_latency = (time.time() - baseline_start) * 1000

                # AVI
                avi_start = time.time()
                avi_response = await self._query_avi(client, row['query'])
                avi_latency = (time.time() - avi_start) * 1000

                # Automatic metrics
                auto_metrics = self.evaluator.evaluate_single_query(
                    query=row['query'],
                    expected_answer=row['expected_answer'],
                    system_response=avi_response['response'],
                    filter_result=avi_response.get('input_filter_result', {}),
                    latency_ms=avi_latency,
                )

                result = {
                    'query_id': row.get('id', f"query_{idx}"),
                    'query': row['query'],
                    'expected_answer': row['expected_answer'],
                    'policy': row.get('policy', ''),
                    'company': row.get('company', ''),
                    'baseline_response': baseline_response,
                    'baseline_latency_ms': baseline_latency,
                    'avi_response': avi_response['response'],
                    'avi_latency_ms': avi_latency,
                    **auto_metrics,
                }

                results.append(result)

        return pd.DataFrame(results)
