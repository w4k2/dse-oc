from core import calculate_metrics
from core import plot_streams_matplotlib
from core import drift_metrics_table_mean
from core import pairs_metrics_multi
from core import plot_streams_mean
from core import plot_radars
from core import plot_streams_bexp

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

stream_sets = []
streams_aliases = []

# streams = []
# directory = "moa/"
# mypath = "results/raw_conf/cs/%s" % directory
# streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
# # print(streams)
#
# stream_sets += [streams]
# streams_aliases += ["moa"]

streams = []
directory = "sl/"
mypath = "results/raw_conf/cs/%s" % directory
streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
# print(streams)

stream_sets += [streams]
streams_aliases += ["sl"]

print(stream_sets[0])

# methods = {
#             "DSO-CNN": "CNN",
#             "DSO-ENN": "ENN",
#             "DSO-RENN": "RENN",
#             "DSO-AKNN": "AKNN",
#             "DSO-CC": "CC",
#             "DSO-RUS": "RUS",
#             "DSO-NCR": "NCR",
#             "DSO-OSS": "OSS",
#             "DSO-TL": "TL",
#             "DSO-NM": "NM",
#             "DSO-IHS": "IHS",
#           }

methods = {
            "DSO-AC": "Aglom",
            "DSO-BC": "Birch",
            "DSO-SC": "Spect",
            "DSO-KM": "KMeans",
            "DSO-MB": "MBKM",
          }
#
# methods = {
#             "DSO-CHS": "CHS",
#             "DSO-DBS": "DBS",
#             "DSO-SHS": "SHS",
#           }

method_names = list(methods.keys())
methods_alias = list(methods.values())


metrics_alias = [
           "Gmean",
           "F-score",
           "Precision",
           "Recall",
           "Specificity",
          ]

metrics = [
           "g_mean",
           "f1_score",
           "precision",
           "recall",
           "specificity",
          ]


experiment_names = [
                    "cs"
                    ]

for streams, streams_alias in zip(stream_sets, streams_aliases):
    # print(streams)
    for experiment_name in experiment_names:
        calculate_metrics(method_names, streams, metrics, experiment_name, recount=True)
        # plot_streams_matplotlib(method_names, streams, metrics, experiment_name, gauss=2, methods_alias=methods_alias, metrics_alias=metrics_alias)

    # pairs_metrics_multi(method_names, streams, metrics, experiment_names, methods_alias=methods_alias, metrics_alias=metrics_alias, streams_alias=streams_alias, title=True)
    plot_radars(method_names, streams, streams_alias, metrics, experiment_name, metrics_alias=metrics_alias, methods_alias=methods_alias)
    # find_best_params(method_names, streams, metrics, experiment_name, metrics_alias=metrics_alias, methods_alias=methods_alias)


# for streams, streams_alias in zip(stream_sets[0:-1], streams_aliases[0:-1]):
#     for experiment_name in experiment_names:
#         calculate_metrics(method_names, streams, drift_metrics, experiment_name, recount=True)
#
#     drift_metrics_table_mean(method_names, streams, drift_metrics, experiment_names, methods_alias=methods_alias, metrics_alias=drift_metrics_alias, streams_alias=streams_alias)
