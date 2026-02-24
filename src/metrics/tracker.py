import pandas as pd


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self, preserve_metrics=False):
        """
        Reset all metrics after epoch end.
 
        Args:
            preserve_metrics (bool): if True, don't reset the metrics
                that are related to evaluation (like MOS, SISDR, etc.)
                and don't reset losses (they should always be reset)
        """
        if preserve_metrics:
            metrics_to_preserve = [
                key for key in self._data.index 
                if any(regime in key for regime in ["_4_8", "_8_16", "_8_24", "_4_16", "_4_24", "_24_48", "_8_48"])
            ]
 
            for col in self._data.columns:
                for key in self._data.index:
                    if key not in metrics_to_preserve:
                        self._data.loc[key, col] = 0
        else:
            for col in self._data.columns:
                self._data.loc[:, col] = 0
 

    def update(self, key, sum, count=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        if key not in self._data.index:
            new_row = pd.DataFrame([[0, 0, 0.0]], index=[key], columns=self._data.columns)
            self._data = pd.concat([self._data, new_row])
        self._data.loc[key, "total"] += sum
        self._data.loc[key, "counts"] += count
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

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
