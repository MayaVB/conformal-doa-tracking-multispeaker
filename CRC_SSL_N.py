import argparse
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
parser.add_argument("--speakers", type=int, default=3, help="Number of speakers")
parser.add_argument("--Kmax", type=int, default=3, help="Maximum K value")
parser.add_argument("--lambda_steps", type=int, default=1000, help="Number of steps for lambda")
parser.add_argument("--significance_levels", type=float, nargs='+', default=[0.1, 0.05], help="Significance levels")
parser.add_argument("--split", type=float, default=0.8, help="Calibration split ratio (used when --calib_path/--test_path are not set)")
parser.add_argument("--calib_path", type=str, default=None, help="Path to calibration .npz file")
parser.add_argument("--test_path", type=str, default=None, help="Path to test .npz file")
args = parser.parse_args()

if (args.calib_path is None) != (args.test_path is None):
    raise ValueError("Provide both --calib_path and --test_path together, or neither.")
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
filename = 'data/npz_output_mobile_mobile_final/Reverb_400_ms_SNR_15_dB/speakers_2.npz'


print(f'Current setup: {model_type} with {speakers} speakers')

def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    r_obj = d['rir_obj'].item()
    r = type('Room', (object,), r_obj)()
    r.xl, r.yl = r.yl, r.xl
    burst_metadata = d['burst_metadata'].tolist() if 'burst_metadata' in d else None
    burst_cfg = d['burst_cfg'].item() if 'burst_cfg' in d else None
    return d['speaker_pos'], d['all_estimated_positions'], d['all_likelihood_maps'], r, burst_metadata, burst_cfg

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

coverage_array, area_array = [], []

for iter in range(num_iterations_effective):
    print(f"Fold {iter}:")
    calib_index, test_index = folds_across_lists[iter][0]

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
    coverage, area = cov_set_obj.test(test_sets=test_index.size,
                                      true_positions=te_spk[test_index, ...],
                                      estimated_positions=te_est[test_index, :speakers, ...],
                                      likelihood_maps=te_maps[test_index, :speakers, ...],
                                      significance_level=significance_level,
                                      test_plot=bool(plot),
                                      burst_metadata=burst_meta_test)

    coverage_array.append(coverage)
    area_array.append(area/37/73*100)  # normalize by sphere area
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

_log_path = f"results_{model_type}_reverb{reverb}_snr{snr}_speakers{speakers}.txt"
with open(_log_path, "w") as _f:
    _f.write(f"Model: {model_type} | Reverb: {reverb} ms | SNR: {snr} dB | Speakers: {speakers}\n")
    _f.write(f"Iterations: {num_iterations} | Seed: {seed}\n\n")
    _f.write(_table_str + "\n")
print(f"Table saved to {_log_path}")
