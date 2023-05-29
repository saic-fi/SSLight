import numpy as np


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    # type (float, float, int, int, int, float, int, float) -> np.ndarray
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def multistep_scheduler(base_value, milestones, epochs, niter_per_ep, gamma=0.1, warmup_epochs=0, start_warmup_value=0):
    # type (float, List[int], int, int, float, int, float) -> np.ndarray
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = []
    cur_lr = base_value
    for i in range(len(milestones)):
        if i == 0:
            nsteps = (milestones[i] - warmup_epochs) * niter_per_ep
        else:
            nsteps = (milestones[i] - milestones[i-1]) * niter_per_ep
        schedule.extend([cur_lr] * nsteps)
        cur_lr *= gamma
    nsteps = (epochs - milestones[-1]) * niter_per_ep
    schedule.extend([cur_lr] * nsteps)
    schedule = np.array(schedule)
    assert len(schedule) == epochs * niter_per_ep
    return schedule