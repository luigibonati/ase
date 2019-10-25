from subprocess import Popen


class Launcher:
    def __init__(self, executable, args=tuple(), cores=1,
                 stdout=None, stdin=None, stderr=None,
                 parallel_args=tuple()):
        self.executable = executable
        for thing in [args, parallel_args]:
            if isinstance(thing, str):
                raise TypeError('Please pass lists, not strings; got {!r}'
                                .format(thing))
        self.args = list(args)
        self.parallel_args = list(parallel_args)

        self.stdout = stdout
        self.stdin = stdin
        self.stderr = stderr
        self.cores = int(cores)

    def __repr__(self):
        kws = ', '.join('{}={!r}'.format(*item) for item in vars(self).items())
        return 'Launcher({})'.format(kws)

    def getargs(self):
        in_args = self.parallel_args + [self.executable] + self.args
        out_args = []
        for arg in in_args:
            out_args.append(arg.format(**vars(self)))
        return out_args

    def as_shell_command(self):
        args = self.getargs()

        if self.stdin is not None:
            args += ['<', self.stdin]

        # If we redirect 2>, we also want to write 1> for clarity:
        stdout_redirect = '>' if self.stderr is None else '1>'
        if self.stdout is not None:
            args += [stdout_redirect, self.stdout]

        if self.stderr is not None:
            args += ['2>', self.stderr]

        cmd = ' '.join(args)
        return cmd

    def popen(self, cwd=None):
        from subprocess import Popen
        args = self.getargs()

        opened = {}
        def openfile(name, mode):
            if name is None:
                return None

            # If stdout == stderr, don't open it twice:
            if name in opened:
                return opened[name]
            fd = open(name, mode)
            opened[name] = fd
            return fd

        try:
            proc = Popen(args,
                         stdin=openfile(self.stdin, 'rb'),
                         stdout=openfile(self.stdout, 'wb'),
                         stderr=openfile(self.stderr, 'wb'),
                         cwd=cwd)
        finally:
            for stream in opened.values():
                stream.close()
        return proc
