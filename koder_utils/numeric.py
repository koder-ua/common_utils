from __future__ import annotations

import math
import logging
import itertools
from dataclasses import dataclass
from typing import List, Optional, Callable, Iterable, cast, Tuple, Dict, Any

import numpy
from scipy import stats, optimize
from numpy import linalg
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.stats.mstats_basic import NormaltestResult

from . import Storable, round_digits, NumVector1d, NumVector2d, Number
from .selectored_storage import TimeSeries
from .inumeric import Numpy2d, Numpy1d


logger = logging.getLogger("utils")


DOUBLE_DELTA = 1e-8
MIN_VALUES_FOR_CONFIDENCE = 7
MIN_VAL = 1
MAX_LIN_DIFF = 100
UPPER_ROUND_COEF = 0.99999
NO_VALUE = -1


average = numpy.mean
dev = lambda x: math.sqrt(numpy.var(x, ddof=1))


@dataclass
class ArrayData:
    header: List[str]
    histo_bins: Optional[numpy.ndarray]
    data: Optional[numpy.ndarray]


def auto_edges(vals: Numpy1d, log_base: float = 2, bins: int = 20,
               round_base: int = 10, log_space: bool = True) -> Numpy1d:
    lower = numpy.min(vals)
    upper = numpy.max(vals)
    return auto_edges2(lower, upper, log_base, bins, round_base, log_space=log_space)


def auto_edges2(lower: float, upper: float, log_base: float = 2,
                bins: int = 20, round_base: int = 10, log_space: bool = True) -> Numpy1d:
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


def approximate_ts(times: NumVector1d, values: NumVector1d, begin: float, end: float, step: float = 1000000) -> Numpy1d:
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
def prepare_heatmap(data: NumVector2d, bins_vals: NumVector1d,
                    bins_count: int, outliers_perc: Tuple[float, float]) -> Tuple[Numpy2d, Numpy1d]:
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


class StatProps(Storable):
    """Statistic properties for timeseries with unknown data distribution"""

    __ignore_fields__ = ['data']

    def __init__(self, data: numpy.array, units: str) -> None:
        self.perc_99: Optional[float] = None
        self.perc_95: Optional[float] = None
        self.perc_90: Optional[float] = None
        self.perc_50: Optional[float] = None
        self.perc_10: Optional[float] = None
        self.perc_5: Optional[float] = None
        self.perc_1: Optional[float] = None

        self.min: Optional[Number] = None
        self.max: Optional[Number] = None

        # bin_center: bin_count
        self.log_bins = False
        self.bins_populations: Optional[Numpy1d] = None

        # bin edges, one more element that in bins_populations
        self.bins_edges: Optional[Numpy1d] = None

        self.data = data
        self.units = units

    def __str__(self) -> str:
        res = ["{}(size = {}):".format(self.__class__.__name__, len(self.data))]
        for name in ["perc_1", "perc_5", "perc_10", "perc_50", "perc_90", "perc_95", "perc_99"]:
            res.append("    {} = {}".format(name, round_digits(getattr(self, name))))
        res.append("    range {} {}".format(round_digits(self.min), round_digits(self.max)))
        return "\n".join(res)

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        data = super().raw()
        data['bins_mids'] = list(data['bins_mids'])
        data['bins_populations'] = list(data['bins_populations'])
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> StatProps:
        data['bins_mids'] = numpy.array(data['bins_mids'])
        data['bins_populations'] = numpy.array(data['bins_populations'])
        return cast(StatProps, super().fromraw(data))


class HistoStatProps(StatProps):
    """Statistic properties for 2D timeseries with unknown data distribution and histogram as input value.
    Used for latency"""
    def __init__(self, data: numpy.array, units: str) -> None:
        StatProps.__init__(self, data, units)


class NormStatProps(StatProps):
    """Statistic properties for timeseries with normal data distribution. Used for iops/bw"""
    def __init__(self, data: numpy.array, units: str) -> None:
        StatProps.__init__(self, data, units)
        self.average: Optional[float] = None
        self.deviation: Optional[float] = None
        self.confidence: Optional[float] = None
        self.confidence_level: Optional[float] = None
        self.normtest: Optional[NormaltestResult] = None
        self.skew: Optional[float] = None
        self.kurt: Optional[float] = None

    def __str__(self) -> str:
        res = ["NormStatProps(size = {}):".format(len(self.data)),
               "    distr = {} ~ {}".format(round_digits(self.average), round_digits(self.deviation)),
               "    confidence({0.confidence_level}) = {1}".format(self, round_digits(self.confidence)),
               "    perc_1 = {}".format(round_digits(self.perc_1)),
               "    perc_5 = {}".format(round_digits(self.perc_5)),
               "    perc_10 = {}".format(round_digits(self.perc_10)),
               "    perc_50 = {}".format(round_digits(self.perc_50)),
               "    perc_90 = {}".format(round_digits(self.perc_90)),
               "    perc_95 = {}".format(round_digits(self.perc_95)),
               "    perc_99 = {}".format(round_digits(self.perc_99)),
               "    range {} {}".format(round_digits(self.min), round_digits(self.max)),
               "    normtest = {0.normtest}".format(self),
               "    skew ~ kurt = {0.skew} ~ {0.kurt}".format(self)]
        return "\n".join(res)

    def raw(self) -> Dict[str, Any]:
        data = super().raw()
        data['normtest'] = (data['nortest'].statistic, data['nortest'].pvalue)
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'NormStatProps':
        data['normtest'] = NormaltestResult(*data['normtest'])
        return cast(NormStatProps, super().fromraw(data))


def calc_norm_stat_props(ts: TimeSeries, bins_count: int = None, confidence: float = 0.95) -> NormStatProps:
    """"Calculate statistical properties of array of numbers"""

    res = NormStatProps(ts.data, ts.units)  # type: ignore

    if len(ts.data) == 0:
        raise ValueError("Input array is empty")

    res.average = average(ts.data)
    res.deviation = dev(ts.data)

    data = sorted(ts.data)
    res.max = data[-1]
    res.min = data[0]
    pcs = numpy.percentile(data, q=[1.0, 5.0, 10., 50., 90., 95., 99.])
    res.perc_1, res.perc_5, res.perc_10, res.perc_50, res.perc_90, res.perc_95, res.perc_99 = pcs

    if len(data) >= MIN_VALUES_FOR_CONFIDENCE:
        res.confidence = stats.sem(ts.data) * \
                         stats.t.ppf((1 + confidence) / 2, len(ts.data) - 1)
        res.confidence_level = confidence
    else:
        res.confidence = None
        res.confidence_level = None

    if bins_count is not None:
        res.bins_populations, res.bins_edges = numpy.histogram(ts.data, bins=bins_count)
        res.bins_edges = res.bins_edges[:-1]

    try:
        res.normtest = stats.mstats.normaltest(ts.data)
    except Exception as exc:
        logger.warning("stats.mstats.normaltest failed with error: %s", exc)

    res.skew = stats.skew(ts.data)
    res.kurt = stats.kurtosis(ts.data)

    return res


# update this code
def rebin_histogram(bins_populations: numpy.array,
                    bins_edges: numpy.array,
                    new_bins_count: int,
                    left_tail_idx: int = None,
                    right_tail_idx: int = None,
                    log_bins: bool = False) -> Tuple[numpy.array, numpy.array]:
    # rebin large histogram into smaller with new_bins bins, linearly distributes across
    # left_tail_idx:right_tail_idx range

    assert len(bins_populations.shape) == 1
    assert len(bins_edges.shape) == 1
    assert bins_edges.shape[0] == bins_populations.shape[0]

    if left_tail_idx is None:
        min_val = bins_edges[0]
    else:
        min_val = bins_edges[left_tail_idx]

    if right_tail_idx is None:
        max_val = bins_edges[-1]
    else:
        max_val = bins_edges[right_tail_idx]

    if log_bins:
        assert min_val > 1E-3
        step = (max_val / min_val) ** (1 / new_bins_count)
        new_bins_edges = min_val * (step ** numpy.arange(new_bins_count))  # type: numpy.array
    else:
        new_bins_edges = numpy.linspace(min_val, max_val, new_bins_count + 1, dtype='float')[:-1]  # type: numpy.array

    old_bins_pos = numpy.searchsorted(new_bins_edges, bins_edges, side='right')
    new_bins = numpy.zeros(new_bins_count, dtype=int)  # type: numpy.array

    # last source bin can't be split
    # TODO: need to add assert for this
    new_bins[-1] += bins_populations[-1]
    bin_sizes = bins_edges[1:] - bins_edges[:-1]

    # correct position to get bin idx from edge idx
    old_bins_pos -= 1
    old_bins_pos[old_bins_pos < 0] = 0
    new_bins_sizes = new_bins_edges[1:] - new_bins_edges[:-1]

    for population, begin, end, bsize in zip(bins_populations[:-1], old_bins_pos[:-1], old_bins_pos[1:], bin_sizes):
        if begin == end:
            new_bins[begin] += population
        else:
            density = population / bsize
            for curr_box in range(begin, end):
                cnt = min(int(new_bins_sizes[begin] * density + 0.5), population)
                new_bins[begin] += cnt
                population -= cnt

    return new_bins, new_bins_edges


def calc_histo_stat_props(ts: TimeSeries,
                          bins_edges: numpy.array = None,
                          rebins_count: int = None,
                          tail: float = 0.005) -> HistoStatProps:
    if bins_edges is None:
        bins_edges = ts.histo_bins

    res = HistoStatProps(ts.data, ts.units)

    # summ across all series
    aggregated = ts.data.sum(axis=0, dtype='int')
    total = aggregated.sum()

    # percentiles levels
    expected = list(numpy.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]) * total)
    cumsum = numpy.cumsum(aggregated)

    percentiles_bins = numpy.searchsorted(cumsum, expected)
    percentiles = bins_edges[percentiles_bins]
    res.perc_1, res.perc_5, res.perc_10, res.perc_50, res.perc_90, res.perc_95, res.perc_99 = percentiles

    # don't show tail ranges on histogram
    left_tail_idx, right_tail_idx = numpy.searchsorted(cumsum, [tail * total, (1 - tail) * total])

    # minimax and maximal non-zero elements
    non_zero = numpy.nonzero(aggregated)[0]
    if len(non_zero) > 0:
        res.min = bins_edges[aggregated[non_zero[0]]]
        res.max = bins_edges[non_zero[-1] + (1 if non_zero[-1] != len(bins_edges) - 1 else 0)]
    else:
        res.min = res.max = 0

    res.log_bins = False
    if rebins_count is not None:
        res.bins_populations, res.bins_edges = rebin_histogram(aggregated, bins_edges, rebins_count,
                                                               left_tail_idx, right_tail_idx)
    else:
        res.bins_populations = aggregated
        res.bins_edges = bins_edges.copy()

    return res


def groupby_globally(data: Iterable, key_func: Callable):
    grouped = {}  # type: ignore
    grouped_iter = itertools.groupby(data, key_func)

    for (bs, cache_tp, act, conc), curr_data_it in grouped_iter:
        key = (bs, cache_tp, act, conc)
        grouped.setdefault(key, []).extend(curr_data_it)

    return grouped


def approximate_curve(x: List[Number], y: List[float], xnew: List[Number], curved_coef: int) -> List[float]:
    """returns ynew - y values of some curve approximation"""
    return cast(List[float], chebval(xnew, chebfit(x, y, curved_coef)))


def approximate_line(x: List[Number], y: List[float], xnew: List[Number], relative_dist: bool = False) -> List[float]:
    """
    x, y - test data, xnew - dots, where we want find approximation
    if not relative_dist distance = y - newy
    returns ynew - y values of linear approximation
    """
    ox = numpy.array(x)
    oy = numpy.array(y)

    # set approximation function
    def func_line(tpl, x):
        return tpl[0] * x + tpl[1]

    def error_func_rel(tpl, x, y):
        return 1.0 - y / func_line(tpl, x)

    def error_func_abs(tpl, x, y):
        return y - func_line(tpl, x)

    # choose distance mode
    error_func = error_func_rel if relative_dist else error_func_abs

    tpl_initial = tuple(linalg.solve([[ox[0], 1.0], [ox[1], 1.0]],
                                     oy[:2]))

    # find line
    tpl_final, success = optimize.leastsq(error_func, tpl_initial[:], args=(ox, oy))

    # if error
    if success not in range(1, 5):
        raise ValueError("No line for this dots")

    # return new dots
    return func_line(tpl_final, numpy.array(xnew))


def moving_average(data: numpy.array, window: int) -> numpy.array:
    cumsum = numpy.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window


def moving_dev(data: numpy.array, window: int) -> numpy.array:
    cumsum = numpy.cumsum(data)
    cumsum2 = numpy.cumsum(data ** 2)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    cumsum2[window:] = cumsum2[window:] - cumsum2[:-window]
    return ((cumsum2[window - 1:] - cumsum[window - 1:] ** 2 / window) / (window - 1)) ** 0.5


def outlier_vals(data: numpy.array, center_range: Tuple[int, int], cut_range: float) -> Tuple[float, float]:
    v1, v2 = numpy.percentile(data, center_range)
    return (v1 + v2) / 2, (v2 - v1) / 2 * cut_range


def find_ouliers(data: numpy.array, center_range: Tuple[int, int] = (25, 75), cut_range: float = 3.0) -> numpy.array:
    center, rng = outlier_vals(data, center_range, cut_range)
    return numpy.abs(data - center) > rng


def find_ouliers_ts(data: numpy.array,
                    windows_size: int = 30,
                    center_range: Tuple[int, int] = (25, 75),
                    cut_range: float = 3.0) -> numpy.array:
    outliers = numpy.zeros(data.shape, dtype=bool)

    if len(data) < windows_size:
        return outliers

    begin_idx = 0
    if len(data) < windows_size * 2:
        end_idx = (len(data) % windows_size) // 2 + windows_size
    else:
        end_idx = len(data)

    while True:
        cdata = data[begin_idx: end_idx]
        outliers[begin_idx: end_idx] = find_ouliers(cdata, center_range, cut_range)
        begin_idx = end_idx

        if end_idx == len(data):
            break

        end_idx += windows_size
        if len(data) - end_idx < windows_size:
            end_idx = len(data)

    return outliers


def hist_outliers_nd(bin_populations: numpy.array,
                     bin_centers: numpy.array,
                     center_range: Tuple[int, int] = (25, 75),
                     cut_range: float = 3.0) -> Tuple[int, int]:
    assert len(bin_populations) == len(bin_centers)
    total_count = bin_populations.sum()

    perc25 = total_count / 100.0 * center_range[0]
    perc75 = total_count / 100.0 * center_range[1]

    perc25_idx, perc75_idx = numpy.searchsorted(numpy.cumsum(bin_populations), [perc25, perc75])
    middle = (bin_centers[perc75_idx] + bin_centers[perc25_idx]) / 2
    r = (bin_centers[perc75_idx] - bin_centers[perc25_idx]) / 2

    lower_bound = middle - r * cut_range
    upper_bound = middle + r * cut_range

    lower_cut_idx, upper_cut_idx = numpy.searchsorted(bin_centers, [lower_bound, upper_bound])
    return lower_cut_idx, upper_cut_idx


def hist_outliers_perc(bin_populations: numpy.array,
                       bounds_perc: Tuple[float, float] = (0.01, 0.99),
                       min_bins_left: int = None) -> Tuple[int, int]:
    assert len(bin_populations.shape) == 1
    total_count = bin_populations.sum()
    lower_perc = total_count * bounds_perc[0]
    upper_perc = total_count * bounds_perc[1]
    idx1, idx2 = numpy.searchsorted(numpy.cumsum(bin_populations), [lower_perc, upper_perc])

    # don't cut too many bins. At least min_bins_left must left
    if min_bins_left is not None and idx2 - idx1 < min_bins_left:
        missed = min_bins_left - (idx2 - idx1) // 2
        idx2 = min(len(bin_populations), idx2 + missed)
        idx1 = max(0, idx1 - missed)

    return idx1, idx2


def ts_hist_outliers_perc(bin_populations: numpy.array,
                          window_size: int = 10,
                          bounds_perc: Tuple[float, float] = (0.01, 0.99),
                          min_bins_left: int = None) -> Tuple[int, int]:
    assert len(bin_populations.shape) == 2

    points = list(range(0, len(bin_populations), window_size))
    if len(bin_populations) % window_size != 0:
        points.append(points[-1] + window_size)

    ranges: List[Tuple[int, int]] = []
    for begin, end in zip(points[:-1], points[1:]):
        window_hist = bin_populations[begin:end].sum(axis=0)
        ranges.append(hist_outliers_perc(window_hist, bounds_perc=bounds_perc, min_bins_left=min_bins_left))

    return min(i[0] for i in ranges), max(i[1] for i in ranges)


def make_2d_histo(tss: List[TimeSeries],
                  outliers_range: Tuple[float, float] = (0.02, 0.98),
                  bins_count: int = 20,
                  log_bins: bool = False) -> TimeSeries:

    # validate input data
    for ts in tss:
        assert len(ts.times) == len(ts.data), "Time(={}) and data(={}) sizes doesn't equal for {!s}"\
                .format(len(ts.times), len(ts.data), ts.source)
        assert ts.time_units == 's', "All arrays should have the same data units"
        assert ts.units == tss[0].units, "All arrays should have the same data units"
        assert ts.data.shape == tss[0].data.shape, "All arrays should have the same data size"
        assert len(ts.data.shape) == 1, "All arrays should be 1d"

    whole_arr = numpy.concatenate([ts.data for ts in tss])
    whole_arr.shape = [len(tss), -1]

    if outliers_range is not None:
        max_vl, begin, end, min_vl = numpy.percentile(whole_arr,
                                                      [0, outliers_range[0] * 100, outliers_range[1] * 100, 100])
        bins_edges = auto_edges2(begin, end, bins=bins_count, log_space=log_bins)
        fixed_bins_edges = bins_edges.copy()
        fixed_bins_edges[0] = begin
        fixed_bins_edges[-1] = end
    else:
        begin, end = numpy.percentile(whole_arr, [0, 100])
        bins_edges = auto_edges2(begin, end, bins=bins_count, log_space=log_bins)
        fixed_bins_edges = bins_edges

    res_data = numpy.concatenate(numpy.histogram(column, fixed_bins_edges) for column in whole_arr.T)
    res_data.shape = (len(tss), -1)
    res = TimeSeries(data=res_data,
                     times=tss[0].times,
                     units=tss[0].units,
                     source=tss[0].source,
                     time_units=tss[0].time_units,
                     histo_bins=bins_edges)
    return res


def aggregate_histograms(tss: List[TimeSeries],
                         outliers_range: Tuple[float, float] = (0.02, 0.98),
                         bins_count: int = 20,
                         log_bins: bool = False) -> TimeSeries:

    # validate input data
    for ts in tss:
        assert len(ts.times) == len(ts.data), "Need to use stripped time"
        assert ts.time_units == 's', "All arrays should have the same data units"
        assert ts.units == tss[0].units, "All arrays should have the same data units"
        assert ts.data.shape == tss[0].data.shape, "All arrays should have the same data size"
        assert len(ts.data.shape) == 2, "All arrays should be 2d"
        assert ts.histo_bins is not None, "All arrays should be 2d"

    whole_arr = numpy.concatenate([ts.data for ts in tss])
    whole_arr.shape = [len(tss), -1]

    max_val = whole_arr.min()
    min_val = whole_arr.max()

    if outliers_range is not None:
        begin, end = numpy.percentile(whole_arr, [outliers_range[0] * 100, outliers_range[1] * 100])
    else:
        begin = min_val
        end = max_val

    bins_edges = auto_edges2(begin, end, bins=bins_count, log_space=log_bins)

    if outliers_range is not None:
        fixed_bins_edges = bins_edges.copy()
        fixed_bins_edges[0] = begin
        fixed_bins_edges[-1] = end
    else:
        fixed_bins_edges = bins_edges

    res_data = numpy.concatenate(numpy.histogram(column, fixed_bins_edges) for column in whole_arr.T)
    res_data.shape = (len(tss), -1)
    return TimeSeries(res_data,
                      times=tss[0].times,
                      units=tss[0].units,
                      source=tss[0].source,
                      time_units=tss[0].time_units,
                      histo_bins=fixed_bins_edges)
