\# of files = 1, 2
calculation outputs = 0, 1, 2
multiple images = 0, 1, 2

Under the constraint that \# of files is greater than or equal to number of calculation outputs and multiple images. Also, if there are two files, one only cares about the case that both have the same number of images and both have calculator outputs or not. If there is only one file, since this is for printing differences, it must have multiple images. Enumerating:

(1,0,1)
(1,1,1)
(2,0,0)
(2,0,2)
(2,2,0)
(2,2,2)

Support for these use cases is done.
