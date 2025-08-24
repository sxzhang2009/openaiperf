"""
Command-line interface for the ob benchmarking tool.
"""

import click
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="0.2.0")
def main():
    """
    OpenAIPerf Benchmarking Tool (ob)
    
    A standardized, MLPerf-inspired toolkit for running and analyzing
    AI performance benchmarks.
    """
    pass

@main.command()
def list():
    """List all available standardized benchmarks."""
    from ob_tools.registry import list_benchmarks
    # The import below is crucial to trigger benchmark registration
    import ob_tools.benchmarks
    
    benchmarks = list_benchmarks()
    
    if not benchmarks:
        click.echo("No benchmarks are currently registered.")
        return
        
    click.echo("Available benchmarks:")
    for name in sorted(benchmarks.keys()):
        click.echo(f"- {name}")

@main.command()
@click.argument('benchmark_name', type=str)
def run(benchmark_name: str):
    """
    Run a standardized benchmark by its registered name.
    
    Example:
    ob run llm-generation-gpt2-standard
    """
    from ob_tools.registry import get_benchmark
    from ob_tools.benchmark import BenchmarkConfig, benchmark as global_benchmark
    # The import below is crucial to trigger benchmark registration
    import ob_tools.benchmarks
    
    try:
        benchmark_class = get_benchmark(benchmark_name)
    except ValueError as e:
        logger.error(e)
        click.echo(f"Error: {e}")
        click.echo("Use 'ob list' to see all available benchmarks.")
        return

    logger.info(f"Starting benchmark: {benchmark_name}")
    
    # For now, we use a hardcoded default config.
    # In the future, this could be loaded from a file or command-line args.
    if benchmark_name == "llm-summarization-cnn-dailymail":
        config = BenchmarkConfig(
            name="llm.summarization.cnn_dailymail_standard",
            model_id="t5-small", # A smaller model for faster execution
            dataset_id="cnn_dailymail",
            workload_args={
                "max_length": 150,
                "min_length": 40,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            },
            quality_target={
                "rouge1": 0.2  # A reasonable baseline for t5-small on this task
            }
        )
    else:
        logger.error(f"No default configuration found for benchmark: {benchmark_name}")
        click.echo(f"Error: No default configuration for {benchmark_name}")
        return

    # Instantiate the benchmark with its config and the global context
    benchmark_instance = benchmark_class(config, global_benchmark.context)
    
    try:
        benchmark_instance.execute()
        logger.info(f"Benchmark '{benchmark_name}' finished successfully.")
    except Exception as e:
        logger.error(f"Benchmark '{benchmark_name}' failed with an error: {e}", exc_info=True)
        click.echo(f"Benchmark run failed. See logs for details.")

if __name__ == '__main__':
    main()
