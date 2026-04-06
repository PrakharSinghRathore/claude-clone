"""
Training handler — iterative crew improvement through repeated execution.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainingHandler:
    """
    Manages training iterations for crew improvement.
    
    Records inputs, outputs, and metrics for each iteration and
    optionally saves them to a JSONL file for analysis.
    """
    
    def __init__(self, output_dir: str = ".training_output"):
        self.output_dir = output_dir
        self.iterations: List[Dict[str, Any]] = []
    
    def save_iteration(
        self,
        iteration: int,
        inputs: Dict[str, Any],
        output: Dict[str, Any],
    ) -> None:
        """Record the results of a training iteration."""
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "output": output,
        }
        self.iterations.append(record)
        logger.debug("Recorded training iteration %d", iteration)
    
    def save_to_file(self, filename: Optional[str] = None) -> str:
        """
        Save all iterations to a JSONL file.
        
        Args:
            filename: Output filename. Defaults to timestamped name.
        
        Returns:
            The path to the saved file.
        """
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if not filename:
            filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for record in self.iterations:
                f.write(json.dumps(record, default=str) + "\n")
        
        logger.info("Saved %d training iterations to %s", len(self.iterations), filepath)
        return filepath
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all training iterations."""
        if not self.iterations:
            return {"total_iterations": 0}
        
        successful = [i for i in self.iterations if i["output"].get("status") == "success"]
        failed = [i for i in self.iterations if i["output"].get("status") == "error"]
        
        return {
            "total_iterations": len(self.iterations),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.iterations) if self.iterations else 0,
        }
