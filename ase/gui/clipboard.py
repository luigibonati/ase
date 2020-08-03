from ase.io.jsonio import encode, decode


class AtomsClipboard:
    def __init__(self, tk):
        self.tk = tk

    def get_text(self):
        return self.tk.clipboard_get()

    def set_text(self, text):
        self.tk.clipboard_clear()
        self.tk.clipboard_append(text)

    def get_atoms(self):
        text = self.get_text()
        return decode(text)

    def set_atoms(self, atoms):
        json_text = encode(atoms)
        self.set_text(json_text)
