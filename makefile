params:
	python experiment_best_undersampling.py
	python experiment_best_cluster_method.py
	python experiment_best_cluster_metric.py
	python experiment_main_moa_1d.py

main:
	python experiment_main_moa_1d_O.py
	python experiment_main_sl_1d_O.py
	python experiment_main_sl_1d_dynamic_O.py
	python experiment_main_real_O.py
	python experiment_main_moa_1d.py
	python experiment_main_sl_1d.py
	python experiment_main_sl_1d_dynamic.py
	python experiment_main_real.py
