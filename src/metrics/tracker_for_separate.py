class MetricTrackerSeparate:
    """
    Class to aggregate metrics from many batches.
    """
 
    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
        """
        self.writer = writer
        import pandas as pd
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()
 
    def reset(self, preserve_metrics=False):
        """
        Reset all metrics after epoch end.
 
        Args:
            preserve_metrics (bool): if True, don't reset the metrics
                that are related to evaluation (like MOS, SISDR, etc.)
        """
        if preserve_metrics:
            metrics_to_preserve = [
                key for key in self._data.index 
                if "_4_8" in key or "_8_16" in key
            ]
 
            for col in self._data.columns:
                for key in self._data.index:
                    if key not in metrics_to_preserve:
                        self._data.loc[key, col] = 0
        else:
            for col in self._data.columns:
                self._data[col].values[:] = 0
 
    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.
 
        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        if key not in self._data.index:
            self._data = self._data.reindex(
                list(self._data.index) + [key],
                fill_value=0
            )
 
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]
 
        if self.writer is not None:
            self.writer.add_scalar(key, value)

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data.average[key]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return dict(self._data.average)

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()