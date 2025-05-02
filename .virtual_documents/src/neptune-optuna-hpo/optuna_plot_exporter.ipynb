from optuna.visualization import *
from optuna import Study
from optuna import load_study
from pathlib2 import Path
from optuna import get_all_study_names
from optuna.storages import RDBStorage
from optuna.importance import get_param_importances
import plotly.io as pio
import pandas as pd


datadir = Path("../../data/hyperparameter-optimization/")
lstModels=[]


for d in list(datadir.iterdir()):
    if d.is_dir():
        lstModels.append(d)


study_db=Path("../../data/hyperparameter-optimization/study.db")


storage = RDBStorage(f"sqlite:///{study_db.as_posix()}")


storage.get_all_studies()[0].study_name


study_names=get_all_study_names(storage)


# for study in storage.get_all_studies():
#     study_name = study.study_name
#     model_name = study_name.replace("-449", "")
#     loaded_study = Study(study_name, storage)    
    
#     model_image_directory = datadir.joinpath(model_name, "images")
#     if not model_image_directory.exists():
#         model_image_directory.mkdir(exist_ok=True)


#     ser = pd.Series(get_param_importances(loaded_study))
#     top_params = ser.sort_values(ascending=False).nlargest(4)
    
#     plots = {
#         "plot_param_importances": plot_param_importances,
#         "plot_slice": plot_slice,
#         "plot_rank":plot_rank,
#         "plot_parallel_coordinate": plot_parallel_coordinate,
#         "plot_contour": plot_contour,
#         "plot_optimization_history": plot_optimization_history,
#         "plot_edf": plot_edf,
#     }
    
#     for name, plot in plots.items():
#         plot_name = name
#         plot_fpath = model_image_directory.joinpath(f"{model_name.replace(" ", "_").lower()}_{plot_name}")
        
#         if plot_name in ["plot_rank","plot_contour"]:
#             plot_plot = plot(loaded_study,params=top_params.index.to_list(), target_name="max_precision_optimal_recall_score",)
#         else:
#             plot_plot = plot(loaded_study)
        
#         # write to plotly json
#         ppj = pio.from_json(plot_plot.to_json())

#         ppj.update_layout(template="seaborn")
        
#         # pp.update_layout(font=dict(family="Courier New", size=16, color="lightyellow"),
#             # paper_bgcolor="black",
#             # plot_bgcolor="black",
#         # )

#         ppj.write_image(file=plot_fpath.with_suffix(".png").as_posix(), format="png", scale=3)
#         ppj.write_html(file=plot_fpath.with_suffix(".html").as_posix(),auto_open=True,auto_play=True,full_html=True,include_plotlyjs=True)


for name in storage.get_all_studies():
    print(name.study_name)


study_name = "Logistic Regression-449"
model_name = study_name.replace("-449", "")
loaded_study = Study(study_name, storage)
figures = []

ser = pd.Series(get_param_importances(loaded_study))
top_params = ser.sort_values(ascending=False).nlargest(4)

plots = {
    "plot_param_importances": plot_param_importances,
    "plot_slice": plot_slice,
    "plot_parallel_coordinate": plot_parallel_coordinate,
    "plot_rank":plot_rank,
    "plot_contour": plot_contour,
    "plot_edf": plot_edf,
    "plot_optimization_history": plot_optimization_history    
}

for name, plot in plots.items():
    plot_name = name
    # plot_fpath = model_image_directory.joinpath(f"{model_name.replace(" ", "_").lower()}_{plot_name}")
    
    if plot_name in ["plot_rank", "plot_contour"]:
        plot_plot = plot(loaded_study,params=top_params.index.to_list(), target_name="max_precision_optimal_recall_score",)
    else:
        plot_plot = plot(loaded_study)
    
    # write to plotly json
    ppj = pio.from_json(plot_plot.to_json())

    ppj.update_layout(template="seaborn")
    figures.append(ppj)
    
    # pp.update_layout(font=dict(family="Courier New", size=16, color="lightyellow"),
        # paper_bgcolor="black",
        # plot_bgcolor="black",
    # )


#| label: hpo-logreg
#| hide-input:
for p in figures:
    p.show()


study_name = "Random Forest-449"
model_name = study_name.replace("-449", "")
loaded_study = Study(study_name, storage)
figures = []

ser = pd.Series(get_param_importances(loaded_study))
top_params = ser.sort_values(ascending=False).nlargest(4)

plots = {
    "plot_param_importances": plot_param_importances,
    "plot_slice": plot_slice,
    "plot_parallel_coordinate": plot_parallel_coordinate,
    "plot_rank":plot_rank,
    "plot_contour": plot_contour,
    "plot_edf": plot_edf,
    "plot_optimization_history": plot_optimization_history    
}

for name, plot in plots.items():
    plot_name = name
    # plot_fpath = model_image_directory.joinpath(f"{model_name.replace(" ", "_").lower()}_{plot_name}")
    
    if plot_name in ["plot_rank", "plot_contour"]:
        plot_plot = plot(loaded_study,params=top_params.index.to_list(), target_name="max_precision_optimal_recall_score",)
    else:
        plot_plot = plot(loaded_study)
    
    # write to plotly json
    ppj = pio.from_json(plot_plot.to_json())

    ppj.update_layout(template="seaborn")
    figures.append(ppj)
    
    # pp.update_layout(font=dict(family="Courier New", size=16, color="lightyellow"),
        # paper_bgcolor="black",
        # plot_bgcolor="black",
    # )


#| label: hpo-randomforest
#| hide-input:
for p in figures:
    p.show()


study_name = "XGBoost-449"
model_name = study_name.replace("-449", "")
loaded_study = Study(study_name, storage)
figures = []

ser = pd.Series(get_param_importances(loaded_study))
top_params = ser.sort_values(ascending=False).nlargest(4)

plots = {
    "plot_param_importances": plot_param_importances,
    "plot_slice": plot_slice,
    "plot_parallel_coordinate": plot_parallel_coordinate,
    "plot_rank":plot_rank,
    "plot_contour": plot_contour,
    "plot_edf": plot_edf,
    "plot_optimization_history": plot_optimization_history    
}

for name, plot in plots.items():
    plot_name = name
    # plot_fpath = model_image_directory.joinpath(f"{model_name.replace(" ", "_").lower()}_{plot_name}")
    
    if plot_name in ["plot_rank", "plot_contour"]:
        plot_plot = plot(loaded_study,params=top_params.index.to_list(), target_name="max_precision_optimal_recall_score",)
    else:
        plot_plot = plot(loaded_study)
    
    # write to plotly json
    ppj = pio.from_json(plot_plot.to_json())

    ppj.update_layout(template="seaborn")
    figures.append(ppj)
    
    # pp.update_layout(font=dict(family="Courier New", size=16, color="lightyellow"),
        # paper_bgcolor="black",
        # plot_bgcolor="black",
    # )


#| label: hpo-xgboost
#| hide-input:
for p in figures:
    p.show()


study_name = "LightGBM-449"
model_name = study_name.replace("-449", "")
loaded_study = Study(study_name, storage)
figures = []

ser = pd.Series(get_param_importances(loaded_study))
top_params = ser.sort_values(ascending=False).nlargest(4)

plots = {
    "plot_param_importances": plot_param_importances,
    "plot_slice": plot_slice,
    "plot_parallel_coordinate": plot_parallel_coordinate,
    "plot_rank":plot_rank,
    "plot_contour": plot_contour,
    "plot_edf": plot_edf,
    "plot_optimization_history": plot_optimization_history    
}

for name, plot in plots.items():
    plot_name = name
    # plot_fpath = model_image_directory.joinpath(f"{model_name.replace(" ", "_").lower()}_{plot_name}")
    
    if plot_name in ["plot_rank", "plot_contour"]:
        plot_plot = plot(loaded_study,params=top_params.index.to_list(), target_name="max_precision_optimal_recall_score",)
    else:
        plot_plot = plot(loaded_study)
    
    # write to plotly json
    ppj = pio.from_json(plot_plot.to_json())

    ppj.update_layout(template="seaborn")
    figures.append(ppj)
    
    # pp.update_layout(font=dict(family="Courier New", size=16, color="lightyellow"),
        # paper_bgcolor="black",
        # plot_bgcolor="black",
    # )


#| label: hpo-lightgbm
#| hide-input:
for p in figures:
    p.show()





# plot_plot.write_image(file=plot_fpath.with_suffix(".png").as_posix(), format="png")
# plot_plot.write_html(file=plot_fpath.with_suffix(".html").as_posix(), )


# ppj.write_image(file=plot_fpath.with_suffix(".png").as_posix(), format="png", width=800,height=600)
# ppj.write_html(file=plot_fpath.with_suffix(".html").as_posix(),auto_open=True,auto_play=True,full_html=True,include_plotlyjs=True)
