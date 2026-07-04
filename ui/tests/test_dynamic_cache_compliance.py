"""Tests for transformers v4.43+ compliance (DynamicCache API)."""

import pytest
from transformers.cache_utils import DynamicCache


class TestDynamicCacheCompliance:
    """Test that DynamicCache is used correctly in generation."""

    def test_dynamic_cache_import(self):
        """Verify DynamicCache can be imported from transformers."""
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        assert cache is not None

    def test_dynamic_cache_creation(self):
        """Verify DynamicCache can be instantiated."""
        cache = DynamicCache()
        assert hasattr(cache, 'key_cache')
        assert hasattr(cache, 'value_cache')
        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0

    def test_dynamic_cache_in_source(self):
        """Verify DynamicCache is used in the generate method."""
        from src.models.loader import FrozenModelWrapper
        import inspect

        source = inspect.getsource(FrozenModelWrapper.generate)
        assert 'DynamicCache()' in source, \
            "generate() must use DynamicCache()"
        assert 'past_key_values = DynamicCache()' in source, \
            "Must initialize past_key_values as DynamicCache()"

    def test_no_raw_tuple_cache(self):
        """Verify we don't pass raw tuples as past_key_values."""
        from src.models.loader import FrozenModelWrapper
        import inspect

        source = inspect.getsource(FrozenModelWrapper.generate)

        # Verify we're not passing raw tuples
        assert 'past_key_values = ()' not in source, \
            "Do not use raw tuple for past_key_values"
        assert 'past_key_values = []' not in source, \
            "Do not use raw list for past_key_values"
        assert 'past_key_values = tuple' not in source, \
            "Do not use tuple() for past_key_values"

    def test_import_in_loader(self):
        """Verify DynamicCache is imported in loader.py."""
        from src import models
        import inspect

        source = inspect.getsource(models.loader)
        assert 'from transformers.cache_utils import DynamicCache' in source, \
            "DynamicCache must be imported from transformers.cache_utils"

    def test_past_key_values_passed_to_generate(self):
        """Verify past_key_values is passed to model.generate()."""
        from src.models.loader import FrozenModelWrapper
        import inspect

        source = inspect.getsource(FrozenModelWrapper.generate)
        assert 'past_key_values=past_key_values' in source, \
            "model.generate() must receive past_key_values parameter"

    def test_warning_filter_removed(self):
        """Verify the old warning filter is removed."""
        from src import models
        import inspect

        source = inspect.getsource(models.loader)
        assert '_PastKVFilter' not in source, \
            "Old warning filter should be removed"
        assert 'past_key_values' not in source.split('def generate')[0] or \
               'DynamicCache' in source.split('def generate')[0], \
            "Warning suppression should be replaced with DynamicCache"
