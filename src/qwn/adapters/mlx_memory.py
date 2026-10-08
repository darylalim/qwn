"""MLX memory counters for the Registry's memory check."""

import mlx.core as mx


class MlxMemory:
    def active_bytes(self) -> int:
        return mx.get_active_memory()

    def peak_bytes(self) -> int:
        return mx.get_peak_memory()

    def recommended_bytes(self) -> int:
        return int(mx.device_info()["max_recommended_working_set_size"])

    def clear_cache(self) -> None:
        mx.clear_cache()
