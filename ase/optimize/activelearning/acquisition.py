import numpy as np


def acquisition(train_images, candidates, mode='min_energy'):
    """
    Acquisition function class.
    This function is in charge of ordering a list of Atoms candidates to
    be evaluated in order to minimize the number of function queries to get
    to the converged solution.

    Parameters
    ----------
    train_images: list
        List of Atoms containing the previously evaluated structures.
    candidates: list
        Unordered list of Atoms with the candidates to be evaluated.
    mode: string
        Name of the acquisition function in charge of ordering the candidates.
        Modes available are:
            - 'max_energy': Sort candidates by predicted energy (from max. to min.).
            - 'min_energy': Sort candidates by predicted energy (from min to max.).
            - 'max_uncertainty': Sort candidates by predicted uncertainty (from max. to min.).
            - 'min_uncertainty': Sort candidates by predicted uncertainty (from min. to max.).
            - 'random': Randomly shuffles the candidates list.
            - 'max_energy_ucb': Sort candidate by predicted energy plus uncertainty. (from max. to min.)
            - 'max_energy_lcb': Sort candidate by predicted energy minus uncertainty. (from max. to min.)
            - 'min_energy_ucb': Sort candidate by predicted energy plus uncertainty. (from min. to max.)
            - 'min_energy_lcb': Sort candidate by predicted energy minus uncertainty. (from min. to max.)
    Returns
    -------
    Sorted candidates: list
        Ordered list of Atoms with the candidates to be evaluated.
    """

    x = []
    y = []
    pred_x = []
    pred_y = []
    pred_unc = []

    # Gather all required information to decide the order of the candidates.
    for i in candidates:
        pred_x.append(i.get_positions().reshape(-1))
        pred_y.append(i.get_potential_energy())
        pred_unc.append(i.get_calculator().results['uncertainty'])
    for i in train_images:
        x.append(i.get_positions().reshape(-1))
        y.append(i.get_potential_energy())

    implemented_acq = ['min_energy', 'max_energy', 'max_energy_ucb',
                       'max_energy_lcb', 'min_energy_ucb', 'min_energy_lcb',
                       'max_uncertainty', 'min_uncertainty', 'random']

    if mode not in implemented_acq:
        msg = 'The selected acquisition function is not implemented. ' \
              'Implemented are: ' + str(implemented_acq)
        raise NotImplementedError(msg)

    if mode == 'min_energy':
        score_index = np.argsort(pred_y)
    if mode == 'max_energy':
        score_index = list(reversed(np.argsort(pred_y)))

    if mode == 'max_uncertainty':
        score_index = list(reversed(np.argsort(pred_unc)))
    if mode == 'min_uncertainty':
        score_index = np.argsort(pred_unc)

    if mode == 'min_energy_ucb':
        score_index = np.argsort(np.array(pred_y) + np.array(pred_unc))

    if mode == 'min_energy_lcb':
        score_index = np.argsort(np.array(pred_y) - np.array(pred_unc))

    if mode == 'max_energy_ucb':
        e_plus_u = np.array(pred_y) + np.array(pred_unc)
        score_index = list(reversed(np.argsort(e_plus_u)))

    if mode == 'max_energy_ucb':
        e_minus_u = np.array(pred_y) - np.array(pred_unc)
        score_index = list(reversed(np.argsort(e_minus_u)))

    # Order candidates (from best to worst candidates):
    ordered_images = []
    for i in score_index:
        ordered_images.append(candidates[i])

    return ordered_images








