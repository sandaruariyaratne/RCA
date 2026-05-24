"""
AI Performance Intelligent Engine - Fine-tuned LLM Component
This module processes ML model outputs through a fine-tuned Qwen LLM to generate
root cause analysis, confidence scores, evidence, and affected components.
"""

import json
import csv
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict, fields
import subprocess
import sys

# For local Qwen LLM (ollama alternative)
try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


# =====================================================================
# CONFIGURATION
# =====================================================================
# Google Gemini API Key - paste your key here to run cloud analysis easily!
# You can get a free key at: https://aistudio.google.com/
GEMINI_API_KEY = "AIzaSyBErXVMuRBlpCQ8IodDCI5lXbXoiOOv4Ik"
# =====================================================================


@dataclass
class MetricData:
    """Parsed metric data from ML model evidence"""
    error_rate: float
    has_restart: bool
    latency_p95: float
    latency_std: float
    disk_io_rate: float
    memory_usage: float
    restart_flag: float
    cpu_usage_rate: float
    failure_streak: int
    net_throughput: float
    memory_pressure: float
    memory_growth_rate: float
    cpu_container_vs_node_ratio: float


@dataclass
class LLMAnalysisResult:
    """Output structure from LLM analysis"""
    id: str
    application_id: int
    config_id: int
    window_timestamp: str
    ml_severity: str
    ml_root_cause: str
    root_cause: str
    confidence: float
    evidence: str
    affected_component: str
    metrics_summary: Dict[str, Any]
    analysis_reasoning: str
    created_at: str


class MetricsAnalyzer:
    """Analyze metrics and identify thresholds"""
    
    # Default thresholds (can be tuned based on your SLA)
    THRESHOLDS = {
        'latency_p95': 0.05,  # seconds
        'error_rate': 0.01,  # 1%
        'cpu_usage_rate': 0.85,  # 85%
        'memory_usage': 900000000,  # 900MB
        'memory_pressure': 0.8,
        'memory_growth_rate': 50000000,  # 50MB/s growth
        'disk_io_rate': 0.8,
        'cpu_container_vs_node_ratio': 0.9,
        'failure_streak': 5
    }
    
    @staticmethod
    def parse_metrics(evidence_str: str) -> MetricData:
        """Parse evidence JSON string into MetricData"""
        try:
            metrics_dict = json.loads(evidence_str)
            # Filter dict to only include keys that are defined in MetricData
            valid_keys = {f.name for f in fields(MetricData)}
            filtered_dict = {k: v for k, v in metrics_dict.items() if k in valid_keys}
            return MetricData(**filtered_dict)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    
    @staticmethod
    def identify_anomalies(metrics: MetricData) -> Tuple[List[str], Dict[str, float]]:
        """Identify which metrics are anomalous"""
        anomalies = []
        severity_scores = {}
        
        if metrics.latency_p95 > MetricsAnalyzer.THRESHOLDS['latency_p95']:
            anomalies.append('high_latency')
            severity_scores['latency_p95'] = metrics.latency_p95 / MetricsAnalyzer.THRESHOLDS['latency_p95']
        
        if metrics.error_rate > MetricsAnalyzer.THRESHOLDS['error_rate']:
            anomalies.append('high_error_rate')
            severity_scores['error_rate'] = metrics.error_rate / MetricsAnalyzer.THRESHOLDS['error_rate']
        
        if metrics.cpu_usage_rate > MetricsAnalyzer.THRESHOLDS['cpu_usage_rate']:
            anomalies.append('high_cpu_usage')
            severity_scores['cpu_usage_rate'] = metrics.cpu_usage_rate / MetricsAnalyzer.THRESHOLDS['cpu_usage_rate']
        
        if metrics.memory_usage > MetricsAnalyzer.THRESHOLDS['memory_usage']:
            anomalies.append('high_memory_usage')
            severity_scores['memory_usage'] = metrics.memory_usage / MetricsAnalyzer.THRESHOLDS['memory_usage']
        
        if metrics.memory_pressure > MetricsAnalyzer.THRESHOLDS['memory_pressure']:
            anomalies.append('memory_pressure_high')
            severity_scores['memory_pressure'] = metrics.memory_pressure / MetricsAnalyzer.THRESHOLDS['memory_pressure']
        
        if abs(metrics.memory_growth_rate) > MetricsAnalyzer.THRESHOLDS['memory_growth_rate']:
            anomalies.append('memory_leak_indicator')
            severity_scores['memory_growth_rate'] = abs(metrics.memory_growth_rate) / MetricsAnalyzer.THRESHOLDS['memory_growth_rate']
        
        if metrics.disk_io_rate > MetricsAnalyzer.THRESHOLDS['disk_io_rate']:
            anomalies.append('high_disk_io')
            severity_scores['disk_io_rate'] = metrics.disk_io_rate / MetricsAnalyzer.THRESHOLDS['disk_io_rate']
        
        if metrics.cpu_container_vs_node_ratio > MetricsAnalyzer.THRESHOLDS['cpu_container_vs_node_ratio']:
            anomalies.append('cpu_contention')
            severity_scores['cpu_container_vs_node_ratio'] = metrics.cpu_container_vs_node_ratio / MetricsAnalyzer.THRESHOLDS['cpu_container_vs_node_ratio']
        
        if metrics.failure_streak > MetricsAnalyzer.THRESHOLDS['failure_streak']:
            anomalies.append('repeated_failures')
            severity_scores['failure_streak'] = metrics.failure_streak / MetricsAnalyzer.THRESHOLDS['failure_streak']
        
        if metrics.has_restart or metrics.restart_flag > 0:
            anomalies.append('container_restart')
            severity_scores['restart_flag'] = 1.0
        
        return anomalies, severity_scores
    
    @staticmethod
    def correlate_metrics(metrics: MetricData, anomalies: List[str]) -> Dict[str, Any]:
        """Correlate anomalies to identify root causes"""
        correlations = {
            'memory_issue': False,
            'cpu_issue': False,
            'io_issue': False,
            'application_issue': False,
            'infrastructure_issue': False
        }
        
        memory_anomalies = {'high_memory_usage', 'memory_pressure_high', 'memory_leak_indicator'}
        cpu_anomalies = {'high_cpu_usage', 'cpu_contention'}
        io_anomalies = {'high_disk_io'}
        restart_anomalies = {'container_restart', 'repeated_failures'}
        
        if any(a in anomalies for a in memory_anomalies):
            correlations['memory_issue'] = True
        
        if any(a in anomalies for a in cpu_anomalies):
            correlations['cpu_issue'] = True
        
        if any(a in anomalies for a in io_anomalies):
            correlations['io_issue'] = True
        
        if any(a in anomalies for a in restart_anomalies):
            correlations['infrastructure_issue'] = True
        
        if 'high_latency' in anomalies or 'high_error_rate' in anomalies:
            correlations['application_issue'] = True
        
        return correlations


class GeminiLLMAnalyzer:
    """Interface with Google Gemini 2.5 Flash-Lite LLM for root cause analysis"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini LLM analyzer
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.available = bool(self.api_key)
        if not self.available:
            print("Warning: Gemini API Key not found. Please paste your key in the GEMINI_API_KEY variable at the top of this script, set GEMINI_API_KEY environment variable, or pass --gemini-api-key.")
            
    def analyze_with_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Gemini 2.5 Flash-Lite model directly with structured prompt for root cause analysis
        """
        if not self.available:
            return self._synthetic_analysis(context)
            
        prompt = self._build_prompt(context)
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                resp_json = response.json()
                try:
                    result_text = resp_json['candidates'][0]['content']['parts'][0]['text']
                    return self._parse_llm_response(result_text, context)
                except (KeyError, IndexError) as e:
                    print(f"Error parsing Gemini response structure: {e}")
                    return self._synthetic_analysis(context)
            else:
                print(f"Gemini API request failed with status {response.status_code}: {response.text}")
                return self._synthetic_analysis(context)
        except Exception as e:
            print(f"Gemini call failed: {e}")
            return self._synthetic_analysis(context)

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build structured prompt for LLM"""
        metrics = context['metrics']
        anomalies = context['anomalies']
        ml_severity = context['ml_severity']
        ml_root_cause = context['ml_root_cause']
        correlations = context.get('correlations', {})
        
        # Build dynamic guides for the LLM based on telemetry correlations
        recommended_cause = "UNKNOWN"
        recommended_component = "APPLICATION"
        
        if correlations.get('memory_issue'):
            recommended_cause = "MEMORY_LEAK"
            recommended_component = "APPLICATION"
        elif correlations.get('cpu_issue'):
            recommended_cause = "CPU_SATURATION"
            recommended_component = "SCHEDULER"
        elif correlations.get('io_issue'):
            recommended_cause = "IO_BOTTLENECK"
            recommended_component = "STORAGE"
        elif correlations.get('infrastructure_issue'):
            recommended_cause = "CONFIGURATION_ISSUE"
            recommended_component = "SCHEDULER"
        elif 'high_latency' in anomalies or 'high_error_rate' in anomalies:
            recommended_cause = "APPLICATION_BUG"
            recommended_component = "APPLICATION"
            
        prompt = f"""You are an SRE expert. Analyze the following performance outage metrics and provide a root cause analysis:

ML Model Alert context:
- Severity: {ml_severity}
- Initial Root Cause: {ml_root_cause}

Detected Anomalies: {', '.join(anomalies) if anomalies else 'none'}
SRE Telemetry Guides:
- Recommended Root Cause Category: {recommended_cause}
- Recommended Affected Component: {recommended_component}

Container Metrics:
- CPU Usage: {metrics.cpu_usage_rate:.4f}
- Memory Usage: {metrics.memory_usage / 1e9:.2f} GB
- Memory Pressure: {metrics.memory_pressure:.4f}
- Memory Growth Rate: {metrics.memory_growth_rate / 1e6:.2f} MB/s
- Latency P95: {metrics.latency_p95:.4f}s
- Error Rate: {metrics.error_rate:.4f}
- Disk I/O Rate: {metrics.disk_io_rate:.4f}
- Network Throughput: {metrics.net_throughput:.2f} Mbps
- CPU Container/Node Ratio: {metrics.cpu_container_vs_node_ratio:.4f}
- Failure Streak: {metrics.failure_streak}
- Container Restarts: {metrics.has_restart}

Task: Return a raw JSON object containing exactly the following keys:
1. "root_cause": select the root cause (strongly follow the Recommended Root Cause Category: {recommended_cause})
2. "confidence": SRE diagnosis confidence score (float between 0.0 and 1.0)
3. "evidence": brief metric explanation (maximum 15 words, very short)
4. "affected_component": select the component (strongly follow the Recommended Affected Component: {recommended_component})
5. "reasoning": SRE logic summary (maximum 15 words, very short)

Ensure all 5 JSON keys are populated and present in your output. Keep your text values extremely brief to prevent generation loops. Output only valid raw JSON without any other text or code blocks."""
        
        return prompt

    def _parse_llm_response(self, response_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                root_cause = parsed.get('root_cause', 'UNKNOWN')
                confidence = parsed.get('confidence', 0.5)
                try:
                    confidence = min(1.0, max(0.0, float(confidence)))
                except (ValueError, TypeError):
                    confidence = 0.5
                evidence = parsed.get('evidence', '')
                affected_component = parsed.get('affected_component', 'APPLICATION')
                reasoning = parsed.get('reasoning', '')
                
                # --- SRE Guardrails to validate/correct LLM classifications ---
                correlations = context.get('correlations', {})
                metrics = context.get('metrics')
                
                is_invalid_memory_leak = (root_cause in ['MEMORY_LEAK', 'GC_PRESSURE']) and not correlations.get('memory_issue')
                is_invalid_cpu_saturation = (root_cause in ['CPU_SATURATION', 'RESOURCE_CONTENTION']) and not correlations.get('cpu_issue')
                is_invalid_io_bottleneck = (root_cause == 'IO_BOTTLENECK') and not correlations.get('io_issue')
                
                if is_invalid_memory_leak or is_invalid_cpu_saturation or is_invalid_io_bottleneck or root_cause == 'UNKNOWN':
                    # Override overfitted LLM output with correct, metric-driven synthetic analysis
                    fallback = self._synthetic_analysis(context)
                    root_cause = fallback['root_cause']
                    confidence = fallback['confidence']
                    evidence = fallback['evidence']
                    affected_component = fallback['affected_component']
                
                # Enrich generic/repetitive evidence with actual telemetry numbers
                if "Memory leak detected" in evidence or not evidence:
                    if root_cause == 'MEMORY_LEAK' and metrics:
                        evidence = f"Memory pressure is high ({metrics.memory_pressure:.2%}) with memory usage at {metrics.memory_usage/1e9:.2f} GB."
                    elif root_cause == 'GC_PRESSURE' and metrics:
                        evidence = f"Garbage Collection overhead due to high memory pressure ({metrics.memory_pressure:.2%})."
                    elif metrics:
                        evidence = f"Telemetry alert: {root_cause} with p95 latency at {metrics.latency_p95:.4f}s."
                
                # Smart fallback for reasoning if omitted or overridden
                if not reasoning or is_invalid_memory_leak or is_invalid_cpu_saturation or is_invalid_io_bottleneck:
                    anomalies_str = ", ".join(context.get('anomalies', [])) if context.get('anomalies') else "none"
                    reasoning = f"SRE diagnostic analysis of {root_cause}. Detected anomalies: {anomalies_str}."
                    if correlations.get('memory_issue'):
                        reasoning += " Significant memory pressure and growth trends indicate possible heap exhaustion."
                    if correlations.get('cpu_issue'):
                        reasoning += " High CPU usage container-to-node ratio suggests process saturation."
                    if correlations.get('io_issue'):
                        reasoning += " Disk I/O bottlenecks or disk throughput limits may be reached."
                    if not reasoning.endswith('.'):
                        reasoning += "."
                
                # Smart fallback for affected_component if omitted or default
                if affected_component == 'APPLICATION' or not affected_component:
                    if correlations.get('memory_issue'):
                        affected_component = 'APPLICATION'
                    elif correlations.get('cpu_issue'):
                        affected_component = 'SCHEDULER'
                    elif correlations.get('io_issue'):
                        affected_component = 'STORAGE'
                    else:
                        affected_component = 'APPLICATION'
                
                return {
                    'root_cause': root_cause,
                    'confidence': confidence,
                    'evidence': evidence,
                    'affected_component': affected_component,
                    'reasoning': reasoning
                }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
        
        return self._synthetic_analysis(context)

    def _synthetic_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthetic analysis when LLM is not available.
        Uses rule-based logic to simulate LLM output.
        """
        metrics = context['metrics']
        anomalies = context['anomalies']
        correlations = context['correlations']
        
        # Determine root cause based on correlations and anomalies
        if correlations['memory_issue']:
            if abs(metrics.memory_growth_rate) > 100000000:
                root_cause = 'MEMORY_LEAK'
                confidence = 0.85
                component = 'APPLICATION'
                evidence = f"Detected memory growth rate of {metrics.memory_growth_rate/1e6:.1f} MB/s with memory usage at {metrics.memory_usage/1e9:.2f}GB and pressure {metrics.memory_pressure:.2%}"
            else:
                root_cause = 'GC_PRESSURE'
                confidence = 0.75
                component = 'APPLICATION'
                evidence = f"High memory pressure ({metrics.memory_pressure:.2%}) and usage ({metrics.memory_usage/1e9:.2f}GB) causing garbage collection overhead"
        
        elif correlations['cpu_issue']:
            if metrics.cpu_container_vs_node_ratio > 0.8:
                root_cause = 'RESOURCE_CONTENTION'
                confidence = 0.80
                component = 'SCHEDULER'
                evidence = f"CPU container vs node ratio at {metrics.cpu_container_vs_node_ratio:.2%} indicates contention with other workloads"
            else:
                root_cause = 'CPU_SATURATION'
                confidence = 0.75
                component = 'APPLICATION'
                evidence = f"CPU usage at {metrics.cpu_usage_rate:.2%} causing latency increase to {metrics.latency_p95:.4f}s"
        
        elif correlations['io_issue']:
            root_cause = 'IO_BOTTLENECK'
            confidence = 0.70
            component = 'STORAGE'
            evidence = f"Disk I/O rate at {metrics.disk_io_rate:.4f} with high latency P95 of {metrics.latency_p95:.4f}s"
        
        elif correlations['infrastructure_issue']:
            root_cause = 'CONFIGURATION_ISSUE'
            confidence = 0.65
            component = 'SCHEDULER'
            evidence = f"Container restarts detected ({metrics.restart_flag:.1%}) with failure streak of {metrics.failure_streak}"
        
        elif 'high_latency' in anomalies and 'high_error_rate' in anomalies:
            root_cause = 'APPLICATION_BUG'
            confidence = 0.60
            component = 'APPLICATION'
            evidence = f"High latency ({metrics.latency_p95:.4f}s) and error rate ({metrics.error_rate:.2%}) indicate application issue"
        
        else:
            root_cause = 'UNKNOWN'
            confidence = 0.50
            component = 'APPLICATION'
            evidence = f"Detected anomalies: {', '.join(anomalies)}"
        
        reasoning = f"Analysis based on {len(anomalies)} detected anomalies. Correlations: Memory={correlations['memory_issue']}, CPU={correlations['cpu_issue']}, IO={correlations['io_issue']}"
        
        return {
            'root_cause': root_cause,
            'confidence': confidence,
            'evidence': evidence,
            'affected_component': component,
            'reasoning': reasoning
        }


class AIPerformanceAnalysisEngine:
    """Main engine orchestrating the analysis pipeline"""
    
    def __init__(self, use_llm: bool = False, gemini_api_key: str = None):
        """
        Initialize the analysis engine
        
        Args:
            use_llm: Whether to use actual cloud Gemini LLM
            gemini_api_key: OpenRouter API key
        """
        self.metrics_analyzer = MetricsAnalyzer()
        self.provider = 'gemini'
        self.llm_analyzer = GeminiLLMAnalyzer(api_key=gemini_api_key) if use_llm else None
        self.results = []
    
    def process_ml_output(self, ml_record: Dict[str, Any]) -> LLMAnalysisResult:
        """
        Process a single ML model output record
        
        Args:
            ml_record: ML model output row
            
        Returns:
            LLMAnalysisResult with complete analysis
        """
        # Parse metrics from evidence
        metrics = self.metrics_analyzer.parse_metrics(ml_record['evidence'])
        
        if metrics is None:
            return self._create_error_result(ml_record)
        
        # Identify anomalies
        anomalies, severity_scores = self.metrics_analyzer.identify_anomalies(metrics)
        
        # Correlate metrics
        correlations = self.metrics_analyzer.correlate_metrics(metrics, anomalies)
        
        # Build context for LLM
        context = {
            'metrics': metrics,
            'anomalies': anomalies,
            'severity_scores': severity_scores,
            'correlations': correlations,
            'ml_severity': ml_record.get('severity', 'WARNING'),
            'ml_root_cause': ml_record.get('root_cause', 'GENERAL_DEGRADATION'),
        }
        
        # Analyze with LLM (or synthetic)
        if self.llm_analyzer and self.llm_analyzer.available:
            llm_analysis = self.llm_analyzer.analyze_with_llm(context)
        else:
            llm_analysis = self.llm_analyzer._synthetic_analysis(context) if self.llm_analyzer else self._synthetic_analysis(context)
        
        # Create result
        result = LLMAnalysisResult(
            id=ml_record['id'],
            application_id=ml_record['application_id'],
            config_id=ml_record['config_id'],
            window_timestamp=ml_record['window_timestamp'],
            ml_severity=ml_record.get('severity', 'WARNING'),
            ml_root_cause=ml_record.get('root_cause', 'GENERAL_DEGRADATION'),
            root_cause=llm_analysis['root_cause'],
            confidence=llm_analysis['confidence'],
            evidence=llm_analysis['evidence'],
            affected_component=llm_analysis['affected_component'],
            metrics_summary={
                'cpu_usage_rate': round(metrics.cpu_usage_rate, 4),
                'memory_usage_gb': round(metrics.memory_usage / 1e9, 2),
                'memory_pressure': round(metrics.memory_pressure, 4),
                'latency_p95': round(metrics.latency_p95, 4),
                'error_rate': round(metrics.error_rate, 4),
            },
            analysis_reasoning=llm_analysis['reasoning'],
            created_at=datetime.utcnow().isoformat()
        )
        
        return result
    
    def _create_error_result(self, ml_record: Dict[str, Any]) -> LLMAnalysisResult:
        """Create error result when metrics parsing fails"""
        return LLMAnalysisResult(
            id=ml_record['id'],
            application_id=ml_record['application_id'],
            config_id=ml_record['config_id'],
            window_timestamp=ml_record['window_timestamp'],
            ml_severity=ml_record.get('severity', 'WARNING'),
            ml_root_cause=ml_record.get('root_cause', 'GENERAL_DEGRADATION'),
            root_cause='UNKNOWN',
            confidence=0.0,
            evidence='Failed to parse metrics',
            affected_component='APPLICATION',
            metrics_summary={},
            analysis_reasoning='Metrics parsing error',
            created_at=datetime.utcnow().isoformat()
        )
    
    def _synthetic_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback synthetic analysis"""
        return {
            'root_cause': 'UNKNOWN',
            'confidence': 0.5,
            'evidence': 'Using fallback analysis',
            'affected_component': 'APPLICATION',
            'reasoning': 'Synthetic analysis fallback'
        }
    
    def process_file(self, input_csv_path: str, output_csv_path: str, limit: int = None) -> int:
        """
        Process entire CSV file and save results
        
        Args:
            input_csv_path: Path to ML output CSV
            output_csv_path: Path to save LLM analysis results
            limit: Maximum records to process (for testing)
            
        Returns:
            Number of records processed
        """
        records_processed = 0
        
        # Read CSV
        df = pd.read_csv(input_csv_path)
        
        # Limit records if specified
        if limit:
            df = df.head(limit)
        
        print(f"Processing {len(df)} records...")
        
        # Process each record
        for idx, row in df.iterrows():
            try:
                # Apply rate-limiting sleep throttle for cloud Gemini provider (10 RPM free tier limits)
                if idx > 0 and self.provider == 'gemini' and self.llm_analyzer and self.llm_analyzer.available:
                    import time
                    time.sleep(6)
                    
                record_dict = row.to_dict()
                result = self.process_ml_output(record_dict)
                self.results.append(result)
                records_processed += 1
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} records...")
            
            except Exception as e:
                print(f"Error processing record {idx}: {e}")
                continue
        
        # Save results
        self._save_results(output_csv_path)
        print(f"\nResults saved to {output_csv_path}")
        
        return records_processed
    
    def _save_results(self, output_path: str):
        """Save analysis results to CSV"""
        output_data = []
        
        for result in self.results:
            output_data.append({
                'id': result.id,
                'application_id': result.application_id,
                'config_id': result.config_id,
                'window_timestamp': result.window_timestamp,
                'ml_severity': result.ml_severity,
                'ml_root_cause': result.ml_root_cause,
                'root_cause': result.root_cause,
                'confidence': result.confidence,
                'affected_component': result.affected_component,
                'evidence': result.evidence,
                'metrics_summary': json.dumps(result.metrics_summary),
                'analysis_reasoning': result.analysis_reasoning,
                'created_at': result.created_at,
            })
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_path, index=False)
        
        # Also save as JSON for database ingestion
        json_path = output_path.replace('.csv', '.jsonl')
        with open(json_path, 'w') as f:
            for row in output_data:
                f.write(json.dumps(row) + '\n')
        
        print(f"Also saved JSONL format to {json_path}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of analysis"""
        if not self.results:
            return {}
        
        root_causes = [r.root_cause for r in self.results]
        components = [r.affected_component for r in self.results]
        confidences = [r.confidence for r in self.results]
        
        # Convert to native Python types for JSON serialization
        root_cause_dist = {str(k): int(v) for k, v in pd.Series(root_causes).value_counts().items()}
        component_dist = {str(k): int(v) for k, v in pd.Series(components).value_counts().items()}
        
        return {
            'total_records': int(len(self.results)),
            'root_cause_distribution': root_cause_dist,
            'affected_component_distribution': component_dist,
            'avg_confidence': float(round(np.mean(confidences), 3)),
            'min_confidence': float(round(min(confidences), 3)),
            'max_confidence': float(round(max(confidences), 3)),
        }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Performance Analysis Engine')
    parser.add_argument('--input', type=str, default='./MLOutput.csv',
                       help='Input ML output CSV file')
    parser.add_argument('--output', type=str, default='./LLMAnalysisResults.csv',
                       help='Output analysis results CSV file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records to process')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use actual cloud Gemini 2.5 Flash-Lite LLM for analysis')
    parser.add_argument('--gemini-api-key', type=str, default=None,
                       help='Google Gemini API Key (or set GEMINI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = AIPerformanceAnalysisEngine(
        use_llm=args.use_llm, 
        gemini_api_key=args.gemini_api_key
    )
    
    # Process file
    processed = engine.process_file(args.input, args.output, limit=args.limit)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    stats = engine.get_summary_stats()
    print(json.dumps(stats, indent=2))
    
    print(f"\n✓ Successfully processed {processed} records")


if __name__ == '__main__':
    main()
