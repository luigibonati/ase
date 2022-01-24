import os.path
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

url = 'http://wiki.fysik.dtu.dk/ase-files/'


def setup(app):
    pass


for file in ['ase/gui/ag.png',
             'ase/ase-talk.pdf']:
    if os.path.isfile(file):
        continue
    if 1:  # try:
        print(url + os.path.basename(file), file)
        urlretrieve(url + os.path.basename(file), file)
        print('Downloaded:', file)
    # except IOError:
    #     pass
