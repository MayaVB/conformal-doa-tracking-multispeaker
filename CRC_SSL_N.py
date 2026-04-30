import argparse
from scipy.optimize import linear_sum_assignment
from Code.crc_ssl import CoverageSet
from Code.plots import plot_roi_neighbours
from Code.utilities import *

parser = argparse.ArgumentParser(description="Run CRC_SSL_N script with configurable parameters.")
parser.add_argument("--plot", type=int, default=0, help="Enable or disable plotting (0 or 1)")
parser.add_argument("--model_type", type=str, default="srp_dnn", help="Model type (e.g., SRP_PHAT)")
parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations")
parser.add_argument("--seed", type=int, default=1234567890, help="Random seed")
parser.add_argument("--snr", type=int, default=15, help="Signal-to-noise ratio")
parser.add_argument("--reverb", type=int, default=700, help="Reverberation time in ms")
parser.add_argument("--speakers", type=int, default=2, help="Number of speakers")
parser.add_argument("--Kmax", type=int, default=3, help="Maximum K value")
parser.add_argument("--lambda_steps", type=int, default=1000, help="Number of steps for lambda")
parser.add_argument("--significance_levels", type=float, nargs='+', default=[0.1, 0.05], help="Significance levels")
parser.add_argument("--split", type=float, default=0.8, help="Calibration split ratio (used when --calib_path/--test_path are not set)")
parser.add_argument("--calib_path", type=str, default=None, help="Path to calibration .npz file")
parser.add_argument("--test_path", type=str, default=None, help="Path to test .npz file")
parser.add_argument("--tracking_path", type=str, default=None,
                    help="Path to flat tracking .npz file (enables tracking mode; mutually exclusive with --calib_path)")
parser.add_argument("--tracking_calib_path", type=str, default=None,
                    help="Path to flat calib .npz for tracking mode with pre-split files")
parser.add_argument("--tracking_test_path", type=str, default=None,
                    help="Path to flat test .npz for tracking mode with pre-split files")
args = parser.parse_args()

if (args.calib_path is None) != (args.test_path is None):
    raise ValueError("Provide both --calib_path and --test_path together, or neither.")
if (args.tracking_calib_path is None) != (args.tracking_test_path is None):
    raise ValueError("Provide both --tracking_calib_path and --tracking_test_path together, or neither.")
tracking_split_mode = args.tracking_calib_path is not None
if args.tracking_path is not None and (args.calib_path is not None or tracking_split_mode):
    raise ValueError("--tracking_path is mutually exclusive with --calib_path/--test_path and --tracking_calib_path/--tracking_test_path.")
if tracking_split_mode and args.calib_path is not None:
    raise ValueError("--tracking_calib_path/--tracking_test_path is mutually exclusive with --calib_path/--test_path.")
if not (0 < args.split < 1):
    raise ValueError("--split must be strictly between 0 and 1.")

plot = args.plot
model_type = args.model_type
num_iterations = args.num_iterations
seed = args.seed
np.random.seed(seed)

snr = args.snr
reverb = args.reverb
speakers = args.speakers
Kmax = args.Kmax

lambda_list_ext = np.linspace(0., 1., args.lambda_steps) # Lambda
significance_level = np.array(args.significance_levels)


# filename = f'data/{model_type}/Reverb_{reverb}_ms_SNR_{snr}_dB/speakers_{speakers}.npz'
# filename = 'data/npz_output/Reverb_400_ms_SNR_15_dB/speakers_2.npz'
# filename = 'data/npz_output_mobile/Reverb_400_ms_SNR_15_dB/speakers_2.npz'
filename = 'data/npz_output_tracking_no_burst_calib/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz'


print(f'Current setup: {model_type} with {speakers} speakers')

def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    r_obj = d['rir_obj'].item()
    r = type('Room', (object,), r_obj)()
    r.xl, r.yl = r.yl, r.xl
    burst_metadata = d['burst_metadata'].tolist() if 'burst_metadata' in d else None
    burst_cfg = d['burst_cfg'].item() if 'burst_cfg' in d else None
    return d['speaker_pos'], d['all_estimated_positions'], d['all_likelihood_maps'], r, burst_metadata, burst_cfg

tracking_gt_valid_cal = None  # set by tracking branches; None in non-tracking modes
tracking_gt_valid_te  = None

if args.calib_path is not None:
    print(f'[INFO] Using explicit calib/test paths')
    print(f'[INFO] Calibration: {args.calib_path}')
    print(f'[INFO] Test:        {args.test_path}')
    cal_spk, cal_est, cal_maps, cal_rir, _cal_burst_meta, _cal_burst_cfg = _load_npz(args.calib_path)
    te_spk,  te_est,  te_maps,  te_rir,  te_burst_meta,  te_burst_cfg  = _load_npz(args.test_path)
    n_cal, n_test = cal_spk.shape[0], te_spk.shape[0]
    print(f'[INFO] n_cal={n_cal}, n_test={n_test}')
    room = cal_rir
    hard_cal  = min(100, n_cal)
    hard_test = min(400, n_test)
    if n_cal < 10:
        raise ValueError(f"Calibration dataset too small: {n_cal} < 10")
    if n_test < 5:
        raise ValueError(f"Test dataset too small: {n_test} < 5")
    print(f"[INFO] Random subsampling {hard_cal} calib / {hard_test} test per iteration ({num_iterations} iterations)")
    rng = np.random.default_rng(seed)
    folds_across_lists = [
        ((np.sort(rng.choice(n_cal, size=hard_cal, replace=False)),
          np.sort(rng.choice(n_test, size=hard_test, replace=False))),)
        for _ in range(num_iterations)
    ]
    num_iterations_effective = num_iterations
elif tracking_split_mode:
    # --- Tracking mode with pre-split flat NPZ files ---
    # Two separate flat NPZs for calib and test; test order is preserved (no shuffling).
    def _load_flat_npz(path):
        d = np.load(path, allow_pickle=True)
        r_obj = d['rir_obj'].item()
        r = type('Room', (object,), r_obj)()
        r.xl, r.yl = r.yl, r.xl
        burst_meta = [
            {'burst_active': bool(d['burst_active_per_frame'][i]),
             'burst_grid_idx': tuple(d['burst_grid_idx_per_frame'][i])}
            for i in range(len(d['all_likelihood_maps']))
        ]
        gt_valid = d['gt_valid_per_frame'] if 'gt_valid_per_frame' in d else None
        return (d['speaker_pos'], d['all_estimated_positions'], d['all_likelihood_maps'],
                r, burst_meta, d['sample_idx_per_frame'], gt_valid)

    print(f'[INFO] Tracking split mode')
    print(f'[INFO] Tracking calib: {args.tracking_calib_path}')
    print(f'[INFO] Tracking test:  {args.tracking_test_path}')
    cal_spk, cal_est, cal_maps, room, _cal_burst_meta, cal_sample_idx, cal_gt_valid = _load_flat_npz(args.tracking_calib_path)
    te_spk,  te_est,  te_maps,  _,    te_burst_meta,   te_sample_idx,  te_gt_valid  = _load_flat_npz(args.tracking_test_path)
    tracking_gt_valid_cal = cal_gt_valid  # may be None
    tracking_gt_valid_te  = te_gt_valid
    unique_cal_samples  = np.unique(cal_sample_idx)
    unique_test_samples = np.unique(te_sample_idx)
    n_cal_per_fold  = int(args.split * len(unique_cal_samples))
    n_test_per_fold = int((1 - args.split) * len(unique_test_samples))
    rng = np.random.default_rng(seed)
    folds_across_lists = []
    for _ in range(num_iterations):
        cal_ids  = rng.choice(unique_cal_samples,  size=n_cal_per_fold,  replace=False)
        test_ids = rng.choice(unique_test_samples, size=n_test_per_fold, replace=False)
        calib_frames = np.sort(np.where(np.isin(cal_sample_idx, cal_ids))[0])
        test_frames  = np.sort(np.where(np.isin(te_sample_idx,  test_ids))[0])
        folds_across_lists.append(((calib_frames, test_frames),))
    print(f'[INFO] {num_iterations} folds, {n_cal_per_fold}/{len(unique_cal_samples)} calib / {n_test_per_fold}/{len(unique_test_samples)} test trajectories each')
    num_iterations_effective = num_iterations
elif args.tracking_path is not None:
    # --- Tracking / flat-NPZ mode ---
    # Each row in the flat NPZ is one frame-instance; test order is preserved (no shuffling).
    print(f'[INFO] Tracking mode: loading flat NPZ from {args.tracking_path}')
    d_flat = np.load(args.tracking_path, allow_pickle=True)
    flat_maps = d_flat['all_likelihood_maps']            # (N_frames, spk, H, W)
    flat_est  = d_flat['all_estimated_positions']        # (N_frames, spk, 2)
    flat_spk  = d_flat['speaker_pos']                    # (N_frames, spk, 2)
    sample_idx_per_frame     = d_flat['sample_idx_per_frame']      # (N_frames,)
    burst_active_per_frame   = d_flat['burst_active_per_frame']    # (N_frames,)
    burst_grid_idx_per_frame = d_flat['burst_grid_idx_per_frame']  # (N_frames, 2)
    _gt_valid_flat           = d_flat['gt_valid_per_frame'] if 'gt_valid_per_frame' in d_flat else None

    r_obj = d_flat['rir_obj'].item()
    room = type('Room', (object,), r_obj)()
    room.xl, room.yl = room.yl, room.xl

    unique_samples   = np.unique(sample_idx_per_frame)
    n_total_samples  = len(unique_samples)
    n_calib_per_fold = int(n_total_samples * args.split)
    n_test_per_fold  = n_total_samples - n_calib_per_fold

    te_burst_meta = [
        {'burst_active': bool(burst_active_per_frame[i]),
         'burst_grid_idx': tuple(burst_grid_idx_per_frame[i])}
        for i in range(len(flat_maps))
    ]

    cal_spk  = te_spk  = flat_spk
    cal_est  = te_est  = flat_est
    cal_maps = te_maps = flat_maps
    tracking_gt_valid_cal = _gt_valid_flat
    tracking_gt_valid_te  = _gt_valid_flat

    rng = np.random.default_rng(seed)
    folds_across_lists = []
    for _ in range(num_iterations):
        perm         = rng.permutation(unique_samples)
        calib_ids    = perm[:n_calib_per_fold]
        test_ids     = perm[n_calib_per_fold:]
        calib_frames = np.sort(np.where(np.isin(sample_idx_per_frame, calib_ids))[0])
        test_frames  = np.sort(np.where(np.isin(sample_idx_per_frame, test_ids))[0])
        folds_across_lists.append(((calib_frames, test_frames),))
    print(f'[INFO] {num_iterations} folds, {n_calib_per_fold} calib / {n_test_per_fold} test trajectories each')
    num_iterations_effective = num_iterations
else:
    print(f'[INFO] Loading: {filename}')
    data = np.load(filename, allow_pickle=True)
    speaker_pos = data['speaker_pos']
    all_estimated_positions = data['all_estimated_positions']
    all_likelihood_maps = data['all_likelihood_maps']
    room_obj = data['rir_obj'].item()
    room = type('Room', (object,), room_obj)()
    temp = room.xl
    room.xl = room.yl
    room.yl = temp
    te_burst_meta = data['burst_metadata'].tolist() if 'burst_metadata' in data else None
    te_burst_cfg = data['burst_cfg'].item() if 'burst_cfg' in data else None
    total_dataset_size = speaker_pos.shape[0]
    splits = generate_random_splits(total_samples=total_dataset_size,
                                    num_iterations=num_iterations,
                                    calib_size=int(total_dataset_size * args.split),
                                    num_lists=1, random_seed=42)
    folds_across_lists = list(zip(*splits))
    num_iterations_effective = num_iterations
    cal_spk = te_spk = speaker_pos
    cal_est = te_est = all_estimated_positions
    cal_maps = te_maps = all_likelihood_maps

def compute_peak_stealing(test_frames, spk_pos, lm_maps, burst_meta_test, room, K):
    """Per-speaker count of frames where the burst noise peak > true speaker peak."""
    if burst_meta_test is None:
        return None
    ele_grid = room.xl[:, 0]
    azi_grid = room.yl[0]
    nele, nazi = lm_maps.shape[2], lm_maps.shape[3]
    stolen = np.zeros(K, dtype=int)
    total  = np.zeros(K, dtype=int)
    for i, f in enumerate(test_frames):
        bm = burst_meta_test[i]
        if not bm.get('burst_active', False):
            continue
        burst_cells = [(int(ei), int(ai)) for ei, ai in bm['burst_grid_idx']
                       if ei >= 0 and ai >= 0]
        if not burst_cells:
            continue
        gt_pos = spk_pos[f]          # (K, 2)  [ele, azi]
        est_azi = np.array([
            azi_grid[np.unravel_index(np.argmax(lm_maps[f, k]), (nele, nazi))[1]]
            for k in range(K)
        ])
        gt_azi = gt_pos[:, 1]
        cost = np.array([[np.degrees(np.abs(((est_azi[e] - gt_azi[g] + np.pi) % (2*np.pi)) - np.pi))
                          for g in range(K)] for e in range(K)])
        est_idx, gt_idx = linear_sum_assignment(cost)
        for e, g in zip(est_idx, gt_idx):
            true_ei = np.argmin(np.abs(gt_pos[g, 0] - ele_grid))
            true_ai = np.argmin(np.abs(gt_pos[g, 1] - azi_grid))
            lm_burst = max(lm_maps[f, e, bei, bai] for bei, bai in burst_cells)
            total[e]  += 1
            if lm_burst > lm_maps[f, e, true_ei, true_ai]:
                stolen[e] += 1
    return stolen, total


coverage_array, area_array = [], []
burst_area_list, noburst_area_list = [], []
burst_cov_list,  noburst_cov_list  = [], []
ps_stolen_list,  ps_total_list     = [], []

for iter in range(num_iterations_effective):
    print(f"Fold {iter}:")
    calib_index, test_index = folds_across_lists[iter][0]

    if tracking_gt_valid_cal is not None:
        n_cal_frames = len(calib_index)
        n_cal_invalid = int(np.sum(~tracking_gt_valid_cal[calib_index]))
        print(f"  [gt_valid] calib : {n_cal_invalid}/{n_cal_frames} invalid frames "
              f"({100*n_cal_invalid/n_cal_frames:.1f}%)")
    if tracking_gt_valid_te is not None:
        n_te_frames = len(test_index)
        n_te_invalid = int(np.sum(~tracking_gt_valid_te[test_index]))
        print(f"  [gt_valid] test  : {n_te_invalid}/{n_te_frames} invalid frames "
              f"({100*n_te_invalid/n_te_frames:.1f}%)")

    if plot:
        dest_path = create_save_directory(f'{model_type}_fold_{iter}')
    else:
        dest_path = None
    cov_set_obj = CoverageSet(true_position=cal_spk[calib_index, ...],
                              estimated_positions=cal_est[calib_index, :speakers, ...],
                              likelihood_maps=cal_maps[calib_index, :speakers, ...],
                              lambda_list=lambda_list_ext, room=room, path_=dest_path, plot_function=plot_roi_neighbours)
    cov_set_obj.calibrate(plot=False, plot_coverage_set=0)


    #[Folds, Speakers, Significance Levels]
    burst_meta_test = [te_burst_meta[i] for i in test_index] if te_burst_meta is not None else None
    coverage, area, burst_stats = cov_set_obj.test(test_sets=test_index.size,
                                                    true_positions=te_spk[test_index, ...],
                                                    estimated_positions=te_est[test_index, :speakers, ...],
                                                    likelihood_maps=te_maps[test_index, :speakers, ...],
                                                    significance_level=significance_level,
                                                    test_plot=bool(plot),
                                                    burst_metadata=burst_meta_test)

    coverage_array.append(coverage)
    area_array.append(area/37/73*100)  # normalize by sphere area
    if burst_stats is not None:
        burst_area_list.append(burst_stats['burst_area']   / 37 / 73 * 100)
        noburst_area_list.append(burst_stats['noburst_area'] / 37 / 73 * 100)
        burst_cov_list.append(burst_stats['burst_coverage'])
        noburst_cov_list.append(burst_stats['noburst_coverage'])

    ps = compute_peak_stealing(test_index, te_spk, te_maps, burst_meta_test, room, speakers)
    if ps is not None:
        ps_stolen_list.append(ps[0])
        ps_total_list.append(ps[1])
    print(coverage)
    print(area)

print('Final Results')
coverage_array = np.mean(coverage_array, axis=0)
area_array = np.mean(area_array, axis=0)
print(coverage_array)
print(area_array)

print_results(coverage_array=coverage_array,
              area_array=area_array,
              significance_level=significance_level,
              area_unit='% of grid area')

from tabulate import tabulate as _tabulate
import os as _os
_sig = np.atleast_1d(np.asarray(significance_level, dtype=float))
_headers = ["Speaker"]
for _alpha in _sig:
    _target = 1.0 - _alpha
    _headers += [f"{_target:.2f} Coverage", f"{_target:.2f} Area [% of grid area]"]
_rows = []
for _i in range(coverage_array.shape[0]):
    _row_vals = []
    for _j in range(coverage_array.shape[1]):
        _row_vals.append(f"{coverage_array[_i, _j]:.3f}")
        _row_vals.append(f"{area_array[_i, _j]:.3f}")
    _rows.append([f"Speaker {_i+1}"] + _row_vals)
_table_str = _tabulate(_rows, headers=_headers, tablefmt="fancy_grid")

_ref_path = (args.tracking_path or args.tracking_calib_path or args.calib_path or args.test_path or filename)
_path_parts = _ref_path.replace("\\", "/").split("/")
_dataset   = _path_parts[-3] if len(_path_parts) >= 3 else "unknown_dataset"
_condition = _path_parts[-2] if len(_path_parts) >= 2 else "unknown_condition"
_log_path = f"results_{model_type}_{_dataset}_{_condition}_speakers{speakers}.txt"
with open(_log_path, "w") as _f:
    _f.write(f"Model: {model_type} | Reverb: {reverb} ms | SNR: {snr} dB | Speakers: {speakers}\n")
    _f.write(f"Iterations: {num_iterations} | Seed: {seed}\n\n")
    _f.write(_table_str + "\n")
print(f"Table saved to {_log_path}")

# --- Burst vs non-burst breakdown ---
if burst_area_list:
    burst_area_mean   = np.nanmean(burst_area_list,   axis=0)  # (speakers, sig_levels)
    noburst_area_mean = np.nanmean(noburst_area_list, axis=0)
    burst_cov_mean    = np.nanmean(burst_cov_list,    axis=0)
    noburst_cov_mean  = np.nanmean(noburst_cov_list,  axis=0)

    # Peak-stealing rates (per speaker, summed over folds)
    _ps_rates = None
    if ps_stolen_list:
        _ps_stolen = np.sum(ps_stolen_list, axis=0)   # (K,)
        _ps_total  = np.sum(ps_total_list,  axis=0)
        _ps_rates  = np.where(_ps_total > 0, _ps_stolen / _ps_total, np.nan)

    _b_headers = ["Condition / Speaker"]
    for _alpha in _sig:
        _target = 1.0 - _alpha
        _b_headers += [f"{_target:.2f} Coverage", f"{_target:.2f} Area [% grid]"]
    _b_headers += ["Peak-stolen %"]

    _b_rows = []
    for _i in range(burst_area_mean.shape[0]):
        _ps_str = f"{_ps_rates[_i]:.1%}" if _ps_rates is not None else "-"
        for _label, _cov, _area, _extra in [
            (f"Burst     Spk {_i+1}", burst_cov_mean[_i],   burst_area_mean[_i],   _ps_str),
            (f"No-burst  Spk {_i+1}", noburst_cov_mean[_i], noburst_area_mean[_i], "-"),
        ]:
            _rv = []
            for _j in range(len(_sig)):
                _rv += [f"{_cov[_j]:.3f}", f"{_area[_j]:.3f}"]
            _b_rows.append([_label] + _rv + [_extra])

    _b_table_str = _tabulate(_b_rows, headers=_b_headers, tablefmt="fancy_grid")
    print("\nBurst vs Non-burst breakdown (avg over folds):")
    print(_b_table_str)
    _b_log_path = _log_path.replace(".txt", "_burst_breakdown.txt")
    with open(_b_log_path, "w") as _f:
        _f.write(f"Model: {model_type} | Reverb: {reverb} ms | SNR: {snr} dB | Speakers: {speakers}\n")
        _f.write(f"Iterations: {num_iterations} | Seed: {seed}\n\n")
        _f.write(_b_table_str + "\n")
    print(f"Burst breakdown saved to {_b_log_path}")
