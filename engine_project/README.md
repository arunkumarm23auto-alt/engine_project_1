# Engine life monitor

## Run locally

From this folder, install the dependencies and start the dashboard:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app trains a separate HistGradientBoosting regressor for each engine type from the labeled `_1.csv` datasets. The model target is `Engine Health Percent`; the dashboard presents its 0-100 estimate as RUL percent. It does not estimate remaining hours or cycles because those labels are not in the datasets.

Uploads may be CSV or Excel and can contain any one or more recognized sensor columns. The app selects available numeric sensors, retrains for that feature set, reports the most informative available sensor, and scores each row. The `Engine Health Percent` column is used as the training target and is excluded from uploaded prediction inputs. The bundled test data is used when no upload is selected.

The model was compared on the held-out test sets using MAE and R². Histogram Gradient Boosting had the lowest MAE for both CI and SI datasets among the tested tree regressors.