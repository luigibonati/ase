from ase import io
from ase.parallel import parallel_function, parprint
import datetime
import copy


@parallel_function
def dump_experience(atoms, filename, restart):
    if restart is True:
        try:
            prev_atoms = io.read(filename, ':')  # Actively searching.
            if atoms not in prev_atoms:  # Avoid duplicates.
                parprint('Updating images (experiences) pool...')
                new_atoms = prev_atoms + [atoms]
                io.write(filename=filename, images=new_atoms)
        except Exception:
            io.write(filename=filename, images=atoms, append=True)
    if restart is False:
        io.write(filename=filename, images=atoms, append=False)


@parallel_function
def attach_calculator(train_images, calculator, test_images):
    """ When including the train_images we avoid training multiple times the
    same process"""
    calc = copy.deepcopy(calculator)
    calc.update_train_data(train_images)

    for i in test_images:
        i.set_calculator(copy.deepcopy(calc))
        i.get_potential_energy()
    return test_images


@parallel_function
def print_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@parallel_function
def print_cite_min():
    msg = "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------\n"
    msg += "You are using LGPMin. Please cite: \n"
    msg += "[1] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
    msg += "arXiv:1808.08588. https://arxiv.org/abs/1808.08588. \n"
    msg += "[1] M. H. Hansen, J. A. Garrido Torres, P. C. Jennings, "
    msg += "J. R. Boes, O. G. Mamun and T. Bligaard. arXiv:1904.00904. "
    msg += "https://arxiv.org/abs/1904.00904 \n"
    msg += "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------"
    parprint(msg)




