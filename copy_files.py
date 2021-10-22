import os
import sys
from shutil import copyfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


streams = []

# directory = "sl_1d/incremental/"
# mypath = "results/raw_conf/svm/%s" % directory
# streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
#
# directory = "sl_1d/sudden/"
# mypath = "results/raw_conf/svm/%s" % directory
# streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
#
# directory = "sl_1d_dyn/incremental/"
# mypath = "results/raw_conf/svm/%s" % directory
# streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
#
# directory = "sl_1d_dyn/sudden/"
# mypath = "results/raw_conf/svm/%s" % directory
# streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
#
# directory = "moa_1d/incremental/"
# mypath = "results/raw_conf/svm/%s" % directory
# streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
#
# directory = "moa_1d/sudden/"
# mypath = "results/raw_conf/svm/%s" % directory
# streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

directory = "real/"
mypath = "results/raw_conf/svm/%s" % directory
streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]


experiment_name = "svm"

methods = [
                "DSO",
          ]

counter = 0
count_error = 0

for stream_name in streams:
    for clf_name in methods:
        counter += 1
        try:
            src = "results/raw_conf/%s/%s/%s.csv" % (experiment_name, stream_name, clf_name)
            dst = "results/raw_conf/gnb/%s/%s.csv" % (stream_name, clf_name)
            copyfile(src, dst)
        except:
            print("ERROR", src, dst)
            count_error += 1

for stream_name in streams:
    for clf_name in methods:
        counter += 1
        try:
            src = "results/raw_conf/%s/%s/%s.csv" % (experiment_name, stream_name, clf_name)
            dst = "results/raw_conf/knn/%s/%s.csv" % (stream_name, clf_name)
            copyfile(src, dst)
        except:
            print("ERROR", src, dst)
            count_error += 1

for stream_name in streams:
    for clf_name in methods:
        counter += 1
        try:
            src = "results/raw_conf/%s/%s/%s.csv" % (experiment_name, stream_name, clf_name)
            dst = "results/raw_conf/dtc/%s/%s.csv" % (stream_name, clf_name)
            copyfile(src, dst)
        except:
            print("ERROR", src, dst)
            count_error += 1

print("Succesfully in %0.1f" % ((1-count_error/counter)*100) + "%")
