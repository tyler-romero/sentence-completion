import os
import glob
import operator

raw_path = os.path.join('../data', 'MSR', 'Holmes_Training_Data_raw')
processed_path = os.path.join('../data', 'MSR', 'Holmes_Training_Data')
files = glob.glob(os.path.join(raw_path, '*.TXT'))


def process(input_path):
    base = os.path.basename(input_path)
    output_path = os.path.join(processed_path, base)
    print("Writing: {}".format(output_path))
    with open(input_path, 'r', errors='ignore') as raw:
        for line in raw:
            if line.startswith('*END*THE SMALL PRINT!'):
                break
        with open(output_path, 'w') as out:
            for line in raw:
                out.write(line)

for path in files:
    process(path)
