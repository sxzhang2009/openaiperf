from ob_tools.benchmark import RunnableBenchmark, BenchmarkConfig
from ob_tools.registry import register_benchmark
import logging

logger = logging.getLogger(__name__)

@register_benchmark("llm-summarization-cnn-dailymail")
class HuggingFaceSummarizationBenchmark(RunnableBenchmark):
    """
    A standardized benchmark for text summarization using a subset of the
    CNN/DailyMail dataset and evaluating with the ROUGE metric.
    """

    def setup(self):
        """
        Downloads the model, tokenizer, and the CNN/DailyMail dataset.
        It also pre-selects a small, deterministic subset for the benchmark run.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from datasets import load_dataset
            import torch
            
            logger.info(f"Loading model and tokenizer: {self.config.model_id}")
            # We use AutoModelForSeq2SeqLM for summarization tasks
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_id)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            logger.info(f"Using device: {self.device}")
            
            logger.info(f"Loading dataset: {self.config.dataset_id}")
            # Load the validation split and select a small, deterministic subset
            full_dataset = load_dataset(self.config.dataset_id, "3.0.0", split="validation")
            # We use a fixed seed to ensure the subset is always the same
            self.dataset = full_dataset.shuffle(seed=42).select(range(10)) 
            
            logger.info(f"Using a subset of {len(self.dataset)} samples for the benchmark.")
            
            self.context.record_model_info(self.config.model_id, self.model.config.to_dict())
            self.context.record_workload_info({
                **self.config.workload_args,
                "dataset_id": self.config.dataset_id,
                "dataset_subset_size": len(self.dataset),
                "device": self.device
            })

        except ImportError:
            logger.error("Please install 'transformers', 'torch', 'datasets', and 'evaluate' to run this benchmark.")
            raise

    def run(self):
        """Runs the summarization task and stores the generated summaries."""
        if not self.model or not self.tokenizer or not self.dataset:
            logger.error("Benchmark was not set up correctly. Aborting run.")
            return
            
        import torch
        from ob_tools.benchmark import monitor

        self.generated_summaries = []
        self.reference_summaries = []

        with monitor():
            for sample in self.dataset:
                article = sample['article']
                reference_summary = sample['highlights']
                
                logger.info(f"Summarizing article: {article[:100]}...")
                
                self.context.record_inference_start()
                
                inputs = self.tokenizer(article, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
                
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs['input_ids'],
                        **self.config.workload_args
                    )
                
                generated_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # The number of generated tokens is not as straightforward for seq2seq
                # We'll use a placeholder value for now.
                generated_tokens = summary_ids.shape[1]
                
                self.context.record_inference_end(tokens_generated=generated_tokens, request_count=1)
                
                self.generated_summaries.append(generated_summary)
                self.reference_summaries.append(reference_summary)

    def evaluate(self):
        """
        Calculates the ROUGE score for the generated summaries and checks if
        it meets the quality target.
        """
        if not self.generated_summaries or not self.reference_summaries:
            logger.error("No summaries were generated. Skipping evaluation.")
            return

        try:
            import evaluate
            
            rouge_metric = evaluate.load("rouge")
            results = rouge_metric.compute(
                predictions=self.generated_summaries, 
                references=self.reference_summaries
            )
            
            logger.info(f"Evaluation results (ROUGE): {results}")
            
            # Record the quality score in the final report
            # (Assuming run.json can be extended with a quality section)
            
            # Check against the quality target
            if self.config.quality_target:
                target_rouge1 = self.config.quality_target.get("rouge1", 0)
                if results["rouge1"] < target_rouge1:
                    logger.error(f"Quality target not met! ROUGE-1 score {results['rouge1']:.4f} is below target of {target_rouge1:.4f}")
                else:
                    logger.info("Quality target met.")

        except ImportError:
            logger.error("Please install the 'evaluate' and 'rouge_score' libraries to run evaluation.")
            raise




