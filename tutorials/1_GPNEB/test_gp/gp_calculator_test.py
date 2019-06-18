from ase.calculators.gp.calculator import GPCalculator
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import copy

train_images = read('train_images.traj', ':-2')
test_images = read('test_images.traj', '-7:')

e_prev = []
for i in test_images:
    e_prev.append(i.get_potential_energy())
plt.figure()
plt.plot(np.arange(len(e_prev)), e_prev)
plt.show()

calc = GPCalculator(train_images=[])
calc.update_train_data(train_images=train_images)
save_dict = calc.__dict__

###### Iteration 1:
e = []
u = []
for i in test_images:
    atoms = i
    i.set_calculator(calc)
    i.get_calculator().__dict__ = save_dict
    i.get_calculator().calculate_uncertainty = True

    e.append(i.get_potential_energy())
    u.append(i.get_calculator().results['uncertainty'])

plt.figure()
plt.plot(np.arange(len(e)), e)
plt.errorbar(np.arange(len(e)), e, u)
plt.show()

###### Iteration 2:
# Update train:
calc.update_train_data([read('train_images.traj', -1),
                        read('train_images.traj', -2)])

e = []
u = []
for i in test_images:
    atoms = i
    i.set_calculator(copy.deepcopy(calc))
    i.get_calculator().calculate_uncertainty = True
    e.append(i.get_potential_energy())
    u.append(i.get_calculator().results['uncertainty'])


plt.figure()
plt.plot(np.arange(len(e)), e)
plt.errorbar(np.arange(len(e)), e, u)
plt.show()








exit()

