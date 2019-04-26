import math
from typing import List, Tuple

import numpy


MIN_VAL = 1
MAX_LIN_DIFF = 100
UPPER_ROUND_COEF = 0.99999


ndarray1d = numpy.ndarray
ndarray2d = numpy.ndarray


@dataclass
class ArrayData:
    header: List[str]
    histo_bins: Optional[numpy.ndarray]
    data: Optional[numpy.ndarray]


class TimeSeries:
    """Data series from sensor - either system sensor or from load generator tool (e.g. fio)"""

    def __init__(self, data: numpy.ndarray, times: ndarray1d, units: str, time_units: str, source: DataSource,
                 histo_bins: ndarray1d = None) -> None:
        self.units = units
        self.time_units = time_units

        self.times = times
        self.data = data

        self.source = source
        self.histo_bins = histo_bins
        self.sec2ts_coef = unit_conversion_coef_f('s', self.time_units)
        assert len(self.data) == len(self.times)

    def select(self, trange: Tuple[float, float]) -> 'TimeSeries':
        selected = copy.copy(self)
        idx1, idx2 = numpy.searchsorted(self.times, (trange[0] * self.sec2ts_coef, trange[1] * self.sec2ts_coef))
        idx2 = min(idx2 + 2, len(self.times))
        idx1 = max(0, idx1 - 2)
        selected.data = self.data[idx1: idx2]
        selected.times = self.times[idx1: idx2]
        assert len(selected.data) == len(selected.times)
        return selected

    def __str__(self) -> str:
        return "TS(src={}, time_size={}, dshape={}):\n".format(self.source, len(self.times), *self.data.shape)

    def __repr__(self) -> str:
        return str(self)

    def copy(self, no_data: bool = False) -> 'TimeSeries':
        cp = copy.copy(self)

        if not no_data:
            cp.times = self.times.copy()
            cp.data = self.data.copy()

        cp.source = self.source()
        return cp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSeries):
            return False

        o = cast(TimeSeries, other)

        return o.units == self.units and \
               o.time_units == self.time_units and \
               numpy.array_equal(o.data, self.data) and \
               numpy.array_equal(o.times, self.times) and \
               o.source == self.source and \
               ((self.histo_bins is None and o.histo_bins is None) or numpy.array_equal(self.histo_bins, o.histo_bins))


def auto_edges(vals: ndarray1d, log_base: float = 2, bins: int = 20,
               round_base: int = 10, log_space: bool = True) -> ndarray1d:
    lower = numpy.min(vals)
    upper = numpy.max(vals)
    return auto_edges2(lower, upper, log_base, bins, round_base, log_space=log_space)


def auto_edges2(lower: float, upper: float, log_base: float = 2,
                bins: int = 20, round_base: int = 10, log_space: bool = True) -> ndarray1d:
    if lower == upper:
        return numpy.array([lower * 0.9, lower * 1.1])

    if round_base and lower > MIN_VAL:
        lower = round_base ** (math.floor(math.log(lower) / math.log(round_base)))
        upper = round_base ** (math.floor(math.log(lower) / math.log(round_base) + UPPER_ROUND_COEF))

    if lower < MIN_VAL or upper / lower < MAX_LIN_DIFF or not log_space:
        return numpy.linspace(lower, upper, bins + 1)

    lower_lg = math.log(lower) / math.log(log_base)
    upper_lg = math.log(upper) / math.log(log_base)
    return numpy.logspace(lower_lg, upper_lg, bins + 1, base=log_base)


def approximate_ts(times: array1d, values: array1d, begin: float, end: float, step: float = 1000000) -> ndarray1d:
    if len(times) != len(values):
        raise AssertionError("Times and values arrays has different sizes")

    if begin < times[0] or end > times[-1] or end <= begin:
        raise AssertionError("Can't approximate as at least one border is not beelong data range or incorect borders")

    pos1, pos2 = numpy.searchsorted(times, (begin, end))

    # current real data time chunk begin time
    edge_it = iter(times[pos1 - 1: pos2 + 1])

    # current real data value
    val_it = iter(values[pos1 - 1: pos2 + 1])

    # result array, cumulative value per second
    result = numpy.zeros(int(end - begin) // step)
    idx = 0
    curr_summ = 0

    # end of current time slot
    results_cell_ends = begin + step

    # hack to unify looping
    real_data_end = next(edge_it)
    while results_cell_ends <= end:
        real_data_start = real_data_end
        real_data_end = next(edge_it)
        real_val_left = next(val_it)

        # real data "speed" for interval [real_data_start, real_data_end]
        real_val_ps = float(real_val_left) / (real_data_end - real_data_start)

        while real_data_end >= results_cell_ends and results_cell_ends <= end:
            # part of current real value, which is fit into current result cell
            curr_real_chunk = int((results_cell_ends - real_data_start) * real_val_ps)

            # calculate rest of real data for next result cell
            real_val_left -= curr_real_chunk
            result[idx] = curr_summ + curr_real_chunk
            idx += 1
            curr_summ = 0

            # adjust real data start time
            real_data_start = results_cell_ends
            results_cell_ends += step

        # don't lost any real data
        curr_summ += real_val_left

    return result


# data is timeseries of 1D arrays, each array is view on system parts load at come time
# E.G. OSD loads at t0. t0 + 1, t0 + 2, ...
# return 2D heatmap array
def prepare_heatmap(data: array2d, bins_vals: array1d,
                    bins_count: int, outliers_perc: Tuple[float, float]) -> Tuple[array2d, array1d]:
    """
    :param data: list of histograms, one per line
    :param bins_vals: values at center of each bin
    :param bins_count: result bin count for each column
    :param outliers_perc: pair of outliers limits tupically (0.25, 0.75)
    :return:
    """

    assert len(data.shape) == 2
    assert data.shape[1] == len(bins_vals)

    total_hist = data.sum(axis=0)

    # idx1, idx2 = hist_outliers_perc(total_hist, style.outliers_lat)
    idx1, idx2 = ts_hist_outliers_perc(data, bounds_perc=outliers_perc)

    # don't cut too many bins
    min_bins_left = bins_count
    if idx2 - idx1 < min_bins_left:
        missed = min_bins_left - (idx2 - idx1) // 2
        idx2 = min(len(total_hist), idx2 + missed)
        idx1 = max(0, idx1 - missed)

    data = data[:, idx1:idx2]
    bins_vals = bins_vals[idx1:idx2]

    # don't using rebin_histogram here, as we need apply same bins for many arrays
    step = (bins_vals[-1] - bins_vals[0]) / bins_count
    new_bins_edges = numpy.arange(bins_count) * step + bins_vals[0]
    bin_mapping = numpy.clip(numpy.searchsorted(new_bins_edges, bins_vals) - 1, 0, len(new_bins_edges) - 1)

    # map origin bins ranges to heatmap bins, iterate over rows
    cmap = []
    for line in data:
        curr_bins = [0] * bins_count
        for idx, count in zip(bin_mapping, line):
            curr_bins[idx] += count
        cmap.append(curr_bins)

    return numpy.array(cmap), new_bins_edges


def avg_counters(counts: List[int], values: List[float]) -> numpy.ndarray:
    counts_a = numpy.array(counts, dtype=numpy.float32)
    values_a = numpy.array(values, dtype=numpy.float32)

    with numpy.errstate(divide='ignore', invalid='ignore'):  # type: ignore
        avg_vals = (values_a[1:] - values_a[:-1]) / (counts_a[1:] - counts_a[:-1])

    avg_vals[avg_vals == numpy.inf] = NO_VALUE
    avg_vals[numpy.isnan(avg_vals)] = NO_VALUE  # type: ignore

    return avg_vals  # type: ignore
