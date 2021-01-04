# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne import pick_types_forward
from mne.datasets import sample
from mne.time_frequency import fit_iir_model_raw, morlet
from mne.simulation import simulate_sparse_stc, simulate_evoked

###############################################################################
# Load real data as templates
data_path = sample.data_path()
raw = mne.io.Raw(data_path + "/MEG/sample/sample_audvis_raw.fif")
proj = mne.read_proj(data_path + "/MEG/sample/sample_audvis_ecg-proj.fif")
raw.info["projs"] += proj
raw.info["bads"] = ["MEG 2443", "EEG 053"]  # mark bad channels

fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
ave_fname = data_path + "/MEG/sample/sample_audvis-no-filter-ave.fif"
cov_fname = data_path + "/MEG/sample/sample_audvis-cov.fif"

fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info["bads"])

cov = mne.read_cov(cov_fname)

evoked_template = mne.read_evokeds(ave_fname, condition=0, baseline=None)
evoked_template = evoked_template.pick(["meg", "eeg"])

# label_names = ['Aud-lh', 'Aud-rh']
label_names = ["Aud-lh", "Aud-lh"]
labels = [
    mne.read_label(data_path + "/MEG/sample/labels/%s.label" % ln)
    for ln in label_names
]

###############################################################################
# Generate source time courses and the correspond evoked data


class DataFun():
    def __init__(self, stc_data):
        self.stc_data = stc_data
        self.counter = 0

    def __call__(self, times):
        assert self.counter < len(self.stc_data)
        out = self.stc_data[self.counter]
        self.counter += 1
        return out


def simu_meg(nave=150, white=True, seed=None):
    tmin = -0.1
    sfreq = 1000.0  # Hz
    tstep = 1.0 / sfreq
    n_samples = 600
    times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

    # Generate times series from 2 Morlet wavelets
    stc_data = np.zeros((len(labels), len(times)))
    Ws = morlet(sfreq, [3, 10], n_cycles=[1, 1.5])
    stc_data[0][: len(Ws[0])] = np.real(Ws[0])
    stc_data[1][: len(Ws[1])] = np.real(Ws[1])
    stc_data *= 100 * 1e-9  # use nAm as unit

    # time translation
    stc_data[1] = np.roll(stc_data[1], 80)

    stc = simulate_sparse_stc(
        fwd["src"], n_dipoles=2,
        times=times, data_fun=DataFun(stc_data), labels=labels,
        random_state=0
    )

    ###########################################################################
    # Generate noisy evoked data
    picks = mne.pick_types(raw.info, meg=True, exclude="bads")
    if white:
        iir_filter = None  # for white noise
    else:
        iir_filter = fit_iir_model_raw(
            raw, order=5, picks=picks, tmin=60, tmax=180
        )[1]

    evoked = simulate_evoked(
        fwd,
        stc,
        evoked_template.info,
        cov,
        nave=nave,
        iir_filter=iir_filter,
        random_state=seed,
    )
    return evoked


def simu_bimodal_meg(
    nave=300,
    white=True,
    seed=None,
    freqs=[3, 50],
    n_cycles=[1, 1.5],
    phases=[0, 0],
    offsets=[0, 80],
):
    tmin = -0.1
    sfreq = 1000.0  # Hz
    tstep = 1.0 / sfreq
    n_samples = 600
    times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

    # Generate times series from 2 Morlet wavelets
    stc_data = np.zeros((len(labels), len(times)))
    Ws = morlet(sfreq, freqs, n_cycles=n_cycles)
    stc_data[0][: len(Ws[0])] = np.real(Ws[0] * np.exp(1j * phases[0]))
    stc_data[1][: len(Ws[1])] = np.real(Ws[1] * np.exp(1j * phases[1]))
    stc_data *= 100 * 1e-9  # use nAm as unit

    # time translation
    stc_data[0] = np.roll(stc_data[0], offsets[0])
    stc_data[1] = np.roll(stc_data[1], offsets[1])

    stc = simulate_sparse_stc(
        fwd["src"], n_dipoles=2,
        times=times, data_fun=DataFun(stc_data), labels=labels,
        random_state=0
    )

    ###########################################################################
    # Generate noisy evoked data
    picks = mne.pick_types(raw.info, meg=True, exclude="bads")
    if white:
        iir_filter = None  # for white noise
    else:
        iir_filter = fit_iir_model_raw(
            raw, order=5, picks=picks, tmin=60, tmax=180
        )

    evoked = simulate_evoked(
        fwd,
        stc,
        evoked_template.info,
        cov,
        nave=nave,
        iir_filter=iir_filter,
        random_state=seed,
    )

    return evoked
