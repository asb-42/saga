"""Tests for data ingestion service."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open


class TestDataIngestion:
    """Test data ingestion from filesystem."""

    def test_get_checkpoint_info(self):
        from ui.server.data_ingestion import get_checkpoint_info
        info = get_checkpoint_info("alignment")
        assert info["exists"] is True
        assert info["type"] == "alignment"
        assert info["has_final"] is True
        assert info["size_mb"] > 0

    def test_get_checkpoint_info_nonexistent(self):
        from ui.server.data_ingestion import get_checkpoint_info
        info = get_checkpoint_info("nonexistent_model")
        assert info["exists"] is False
        assert info["type"] == "nonexistent_model"

    def test_get_all_checkpoints(self):
        from ui.server.data_ingestion import get_all_checkpoints
        checkpoints = get_all_checkpoints()
        assert len(checkpoints) == 6
        types = [cp["type"] for cp in checkpoints]
        assert "alignment" in types
        assert "router" in types
        assert "autoencoder" in types
        assert "meta_model" in types
        assert "reward_model" in types
        assert "poisoned_qwen" in types

    def test_get_anomaly_threshold(self):
        from ui.server.data_ingestion import get_anomaly_threshold
        threshold = get_anomaly_threshold()
        assert threshold is not None
        assert "tau" in threshold
        assert "empirical_fpr" in threshold
        assert threshold["tau"] > 0

    def test_get_model_configs(self):
        from ui.server.data_ingestion import get_model_configs
        configs = get_model_configs()
        assert "models" in configs
        assert "alignment" in configs
        assert "router" in configs
        assert "evaluation" in configs

    def test_get_poisoning_results(self):
        from ui.server.data_ingestion import get_poisoning_results
        results = get_poisoning_results()
        assert results is not None
        assert "pattern" in results
        assert "combined_recall" in results["pattern"]
        assert "combined_fpr" in results["pattern"]

    def test_get_full_eval_results(self):
        from ui.server.data_ingestion import get_full_eval_results
        results = get_full_eval_results()
        assert results is not None
        assert "single_models" in results

    def test_get_training_runs_from_tensorboard(self):
        from ui.server.data_ingestion import get_training_runs_from_tensorboard
        runs = get_training_runs_from_tensorboard()
        assert isinstance(runs, list)
        assert len(runs) > 0
        for run in runs:
            assert "directory" in run
            assert "event_file" in run
            assert "last_modified" in run

    def test_get_trainer_state(self):
        from ui.server.data_ingestion import get_trainer_state
        state = get_trainer_state("meta_model")
        assert state is not None
        assert "log_history" in state
        assert "global_step" in state

    def test_get_poisoning_meta(self):
        from ui.server.data_ingestion import get_poisoning_meta
        meta = get_poisoning_meta()
        assert meta is not None
        assert "trigger" in meta
        assert meta["trigger"] == "Year: 2024"

    def test_get_reward_model_meta(self):
        from ui.server.data_ingestion import get_reward_model_meta
        meta = get_reward_model_meta()
        assert meta is not None
        assert "base_model" in meta

    def test_get_all_training_metrics(self):
        from ui.server.data_ingestion import get_all_training_metrics
        metrics = get_all_training_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        for m in metrics:
            assert "checkpoint_type" in m

    def test_get_evaluation_summary(self):
        from ui.server.data_ingestion import get_evaluation_summary
        summary = get_evaluation_summary()
        assert "poisoning" in summary
        assert "full_eval" in summary
        assert "anomaly_threshold" in summary
