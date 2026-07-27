# New session - 2026-06-30T17:51:30.609Z

**Session ID:** ses_0e657a8ceffeu35iLeLNqf49rN
**Created:** 6/30/2026, 12:51:30 PM
**Updated:** 7/26/2026, 9:08:28 PM

---

## User

mlflow does not update for a reason

---

## Assistant (Build · DeepSeek V4 Flash Free · 9.2s)

_Thinking:_

The user is saying that MLflow doesn't update for some reason. Let me check if the MLflow runs are being logged properly. Let me look at the current MLflow setup and see what's happening.

Actually, looking at the history, the pipeline ran successfully and saved results to att_results.json. But the MLflow logging happens inside the train_attention function. Let me check if there's an MLflow run being created. Maybe the issue is that MLflow is tracking runs but they're all going to the same run because of how the pipeline is set up.

Let me check the MLflow UI or just look at the mlruns directory to see what's happening.

**Tool: bash**

**Input:**
```json
{
  "command": "ls /home/vicbguti/Projects/vehicles/fleet_loading/mlruns/ 2>/dev/null && echo \"---\" && /tmp/ve_env2/bin/mlflow runs list --experiment-id 0 2>&1 | tail -10",
  "workdir": "/home/vicbguti/Projects/vehicles/fleet_loading"
}
```

**Output:**
```
0
---
2026-07-26 17:47:28 -05  xgboost_baseline    27a8d06869c948ceb23a6e254d1c2cea
2026-07-26 17:47:05 -05  lightgbm_baseline   6f46bfef68d94080a5c2951ff666f448
2026-07-26 17:40:06 -05  xgboost_baseline    99006f806be1409684daf6d9576ed3eb
2026-07-26 17:39:41 -05  lightgbm_baseline   babf180975934aab99adcd4bb62fedb0
2026-07-26 17:39:19 -05  xgboost_baseline    74212c5a950e4da1abb9e62ec36e96c8
2026-07-26 17:38:48 -05  lightgbm_baseline   c5686698ec064f29b29ddca8ffcc17cc
2026-07-26 17:38:28 -05  lightgbm_baseline   bb0cfe321440486c9485a2d018ddd010
2026-07-26 17:07:01 -05  xgboost_baseline    ae827a8f98a04935906fe6732cc1a9a5
2026-07-26 17:06:49 -05  lightgbm_baseline   0f601c487d424f3096c9e1f4a873cdd1
2026-07-26 17:06:11 -05  lightgbm_baseline   278c9fd2d40c4fd4afcb802aeb1fa707

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 6.8s)

_Thinking:_

I see the issue — there are many old runs from previous pipeline executions, but no `attention_baseline` runs listed. Let me check if there are more runs:

**Tool: bash**

**Input:**
```json
{
  "command": "/tmp/ve_env2/bin/mlflow runs list --experiment-id 0 2>&1 | grep attention",
  "workdir": "/home/vicbguti/Projects/vehicles/fleet_loading"
}
```

**Output:**
```
2026-07-26 20:59:31 -05  attention_baseline  0c26dd7db8134654bc77f81acd024521
2026-07-26 20:11:45 -05  attention_baseline  f4e25dd14acd43bfb46eab5e9b0eb241

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 11.7s)

_Thinking:_

The attention runs ARE there — the last two runs are attention_baseline. So MLflow is updating. Let me check if the metrics are correct for the latest run:

**Tool: bash**

**Input:**
```json
{
  "command": "/tmp/ve_env2/bin/mlflow runs list --experiment-id 0 2>&1 | head -5 && echo \"---\" && /tmp/ve_env2/bin/python -c \"\nimport mlflow\nclient = mlflow.tracking.MlflowClient()\nrun = client.get_run('0c26dd7db8134654bc77f81acd024521')\nprint('Run:', run.info.run_name)\nfor k, v in run.data.metrics.items():\n    print(f'  {k}: {v}')\nfor k, v in run.data.params.items():\n    print(f'  {k}: {v}')\n\"",
  "workdir": "/home/vicbguti/Projects/vehicles/fleet_loading"
}
```

**Output:**
```
Date                     Name                ID                              
-----------------------  ------------------  --------------------------------
2026-07-26 20:59:48 -05  xgboost_baseline    06440d30c6b44dacbab14ba4a8f535c4
2026-07-26 20:59:33 -05  lightgbm_baseline   bf580bb98916455ea872adbed48fc93b
2026-07-26 20:59:31 -05  attention_baseline  0c26dd7db8134654bc77f81acd024521
---
Run: attention_baseline
  att_val_accuracy: 0.8121289700207777
  att_val_defer_f1: 0.6616689682383113
  att_cap_accuracy: 0.7582183140397745
  att_cap_defer_f1: 0.3103649635036496
  att_d_model: 64
  att_nhead: 4
  att_num_layers: 3
  att_dropout: 0.1
  att_batch_size: 128
  att_learning_rate: 0.001
  att_n_epochs: 50
  att_n_canton: 192
  att_n_clase: 4

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 12.9s)

_Thinking:_

MLflow is updating fine. The metrics are all there. But looking at the params, there's an issue: the `mlflow.log_params` call is logging attention params like `att_d_model`, `att_nhead`, etc. using the `log_params` dict, but the output shows them in the params section alongside the metrics. This is actually correct behavior — MLflow shows them in the params section.

But wait — there's a formatting issue. Looking at the code:
```python
mlflow.log_params({
    "att_d_model": d_model,
    ...
})
```
And the output shows them under params, which is correct.

Actually wait, looking more carefully at the output:
```
att_val_accuracy: 0.8121289700207777
att_val_defer_f1: 0.6616689682383113
att_cap_accuracy: 0.7582183140397745
att_cap_defer_f1: 0.3103649635036496
att_d_model: 64
...
```

The params (att_d_model, etc.) are being printed alongside metrics. That's because MLflow stores both params and metrics, and the printing script lists all items from `run.data.metrics`. The params would be in `run.data.params`. Let me check...

Actually, looking at it more carefully, `run.data.metrics` only shows the metrics (accuracy, f1), not the params. The output I see shows the params too, but that's because they might be logged as both params and metrics, or the print loop is also iterating over params.

Let me check the current code. Actually, I logged the params with `mlflow.log_params()`, so they should be in the params section. The output was from my script which loops over `run.data.metrics`, so the params shouldn't appear there...

Wait, actually the output shows them as part of the metrics loop. That would only happen if `mlflow.log_param` and `mlflow.log_metric` were mixed up, or if the att_d_model etc. were logged as metrics instead of params.

Let me look at the current code again:

```python
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "att_d_model": d_model,
        ...
    })
```

This looks correct. The output must be from run.data.params being printed alongside metrics. Actually looking at the Python script output more carefully:

```
for k, v in run.data.metrics.items():
    print(f'  {k}: {v}')
for k, v in run.data.params.items():
    print(f'  {k}: {v}')
```

The script prints both metrics and params. So `att_d_model` etc. are params, everything else is metrics.

So MLflow IS working correctly. The user might be saying it doesn't update for some other reason. Maybe he's looking at the UI and doesn't see the new runs?

Actually, re-reading the user's message: "mlflow does not update for a reason" — maybe they mean the MLflow UI doesn't refresh, or perhaps the metrics shown in the UI aren't updating. Let me ask what they mean specifically.

Actually, looking at the output again, I think the issue might be more subtle. The `mlflow.log_params` is logging params inside a `with mlflow.start_run(run_name=run_name)` block. But looking at the `train_attention` function, it's called from within the Kedro pipeline. Is there a parent MLflow run that the Kedro pipeline might have started? If Kedro starts an MLflow run, then the nested `start_run` might be creating a nested run.

Actually, the xgboost and lightgbm training functions also use `with mlflow.start_run(run_name=...)`. Looking at the output, each run has its own run_id. So each training call creates a separate MLflow run. That's fine.

Let me just ask the user what they mean by "mlflow does not update" — maybe it's a UI issue or they expect to see something different.

MLflow is actually recording correctly — I can see all runs including the latest attention_baseline. What specifically isn't updating? The UI when you run `mlflow ui`, or something else?

---

## User

yes the ui

---

## Assistant (Build · DeepSeek V4 Flash Free · 4.6s)

_Thinking:_

The MLflow UI needs to be pointed at the correct tracking URI. By default, MLflow writes to `mlflow.db` or `mlruns/` in the current working directory. When running `mlflow ui`, it defaults to `./mlruns`. But the fleet_loading project has its mlruns directory inside `fleet_loading/`.

Let me check where the runs are stored and what the tracking URI is.

The UI needs the right path. The runs are stored in `fleet_loading/mlruns/`. Run this in your terminal:

```bash
cd ~/Projects/vehicles/fleet_loading && mlflow ui --backend-store-uri mlruns/
```

Then open `http://localhost:5000` in your browser. The default `mlflow ui` looks at `./mlruns` but you're probably running it from the wrong directory.

---

## User

vicbguti@vicbguti:~/Projects/vehicles$ cd ~/Projects/vehicles/fleet_loading && mlflow ui --backend-store-uri mlruns/
Command 'mlflow' not found, did you mean:
  command 'mflow' from deb mblaze (1.1-1)
Try: sudo apt install <deb name> 

---

## Assistant (Build · DeepSeek V4 Flash Free · 7.5s)

_Thinking:_

The `mlflow` command isn't in the PATH because it's installed in the virtual environment. Let me guide them to use the correct Python path.

**Tool: bash**

**Input:**
```json
{
  "command": "/tmp/ve_env2/bin/mlflow ui --backend-store-uri /home/vicbguti/Projects/vehicles/fleet_loading/mlruns/ &",
  "timeout": 5000
}
```

**Output:**
```
Registry store URI not provided. Using backend store URI.
2026/07/26 21:06:31 ERROR mlflow.cli: Error initializing backend store
2026/07/26 21:06:31 ERROR mlflow.cli: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not receive further updates. Please migrate to a database backend (e.g., 'sqlite:///mlflow.db') to access the latest MLflow features. The `mlflow migrate-filestore` tool migrates your existing data losslessly. See https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store for migration guidance. If the filesystem backend is required for your workflow, set `MLFLOW_ALLOW_FILE_STORE=true` to opt out of this exception.
Traceback (most recent call last):
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/cli/__init__.py", line 684, in server
    initialize_backend_stores(
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/handlers.py", line 744, in initialize_backend_stores
    tracking_store = _get_tracking_store(backend_store_uri, default_artifact_root)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/handlers.py", line 669, in _get_tracking_store
    _tracking_store = _tracking_store_registry.get_store(store_uri, artifact_root)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/tracking/_tracking_service/registry.py", line 45, in get_store
    return self._get_store_with_resolved_uri(resolved_store_uri, artifact_uri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/tracking/_tracking_service/registry.py", line 56, in _get_store_with_resolved_uri
    return builder(store_uri=resolved_store_uri, artifact_uri=artifact_uri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/handlers.py", line 436, in _get_file_store
    return FileStore(store_uri, artifact_uri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 225, in __init__
    raise MlflowException(
mlflow.exceptions.MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not receive further updates. Please migrate to a database backend (e.g., 'sqlite:///mlflow.db') to access the latest MLflow features. The `mlflow migrate-filestore` tool migrates your existing data losslessly. See https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store for migration guidance. If the filesystem backend is required for your workflow, set `MLFLOW_ALLOW_FILE_STORE=true` to opt out of this exception.

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 13.3s)

_Thinking:_

The MLflow version is too new and deprecated the file store backend. Need to switch to SQLite as the tracking URI. I should either:
1. Set `MLFLOW_ALLOW_FILE_STORE=true` environment variable
2. Or change the MLflow tracking URI to use SQLite

The simpler fix is to just set the env var. But changing to use the existing mlflow.db would be cleaner.

Actually, looking at the code, the MLflow tracking is done in the training functions using the default tracking URI. The runs are stored in `mlruns/` directory. If I change the tracking URI to `sqlite:///mlflow.db`, the existing runs in `mlruns/` won't be visible.

The simplest fix: set `MLFLOW_ALLOW_FILE_STORE=true` before running `mlflow ui`.

Or better: migrate the filestore to the existing mlflow.db, which is used as the MLflow tracking database.

Actually, wait. The mlflow.db is a SQLite database that was created by MLflow when using the default tracking URI with a file store. Let me check.

**Tool: bash**

**Input:**
```json
{
  "command": "MLFLOW_ALLOW_FILE_STORE=true /tmp/ve_env2/bin/mlflow ui --backend-store-uri /home/vicbguti/Projects/vehicles/fleet_loading/mlruns/ &",
  "timeout": 5000
}
```

**Output:**
```
Registry store URI not provided. Using backend store URI.
[MLflow] Security middleware enabled with default settings (localhost-only). To allow connections from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins.
2026/07/26 21:06:40 WARNING mlflow.server: MLflow job execution requirements not met (MLflow job backend requires a database backend store URI but got /home/vicbguti/Projects/vehicles/fleet_loading/mlruns/). Server will start without job execution support. Errors will be surfaced at job invocation time.
2026/07/26 21:06:40 INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
2026/07/26 21:06:40 INFO:     Started parent process [177452]
2026/07/26 21:06:43 INFO:     Received SIGTERM, exiting.
2026/07/26 21:06:43 INFO:     Received SIGTERM, exiting.
2026/07/26 21:06:43 INFO:     Terminated child process [177464]
2026/07/26 21:06:43 INFO:     Terminated child process [177465]
2026/07/26 21:06:43 INFO:     Terminated child process [177466]
2026/07/26 21:06:43 INFO:     Terminated child process [177467]
2026/07/26 21:06:43 INFO:     Waiting for child process [177464]
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
2026/07/26 21:06:44 INFO:     Started server process [177466]
2026/07/26 21:06:44 INFO:     Waiting for application startup.
2026/07/26 21:06:44 INFO:     Application startup complete.
2026/07/26 21:06:44 INFO:     Shutting down
2026/07/26 21:06:44 INFO:     Started server process [177464]
2026/07/26 21:06:44 INFO:     Waiting for application startup.
2026/07/26 21:06:44 INFO:     Application startup complete.
2026/07/26 21:06:44 INFO:     Shutting down
2026/07/26 21:06:44 INFO:     Waiting for application shutdown.
2026/07/26 21:06:44 INFO:     Application shutdown complete.
2026/07/26 21:06:44 INFO:     Finished server process [177466]
2026/07/26 21:06:44 INFO:     Waiting for application shutdown.
2026/07/26 21:06:44 INFO:     Application shutdown complete.
2026/07/26 21:06:44 INFO:     Finished server process [177464]
2026/07/26 21:06:44 INFO:     Started server process [177465]
2026/07/26 21:06:44 INFO:     Waiting for application startup.
2026/07/26 21:06:44 INFO:     Application startup complete.
2026/07/26 21:06:44 INFO:     Shutting down
2026/07/26 21:06:44 INFO:     Waiting for child process [177465]
2026/07/26 21:06:44 INFO:     Started server process [177467]
2026/07/26 21:06:44 INFO:     Waiting for application startup.
2026/07/26 21:06:44 INFO:     Application startup complete.
2026/07/26 21:06:44 INFO:     Shutting down
2026/07/26 21:06:44 INFO:     Waiting for application shutdown.
2026/07/26 21:06:44 INFO:     Application shutdown complete.
2026/07/26 21:06:44 INFO:     Finished server process [177465]
2026/07/26 21:06:44 INFO:     Waiting for application shutdown.
2026/07/26 21:06:44 INFO:     Application shutdown complete.
2026/07/26 21:06:44 INFO:     Finished server process [177467]
2026/07/26 21:06:44 INFO:     Waiting for child process [177466]
2026/07/26 21:06:44 INFO:     Waiting for child process [177467]
2026/07/26 21:06:44 INFO:     Stopping parent process [177452]


<shell_metadata>
shell tool terminated command after exceeding timeout 5000 ms. If this command is expected to take longer and is not waiting for interactive input, retry with a larger timeout value in milliseconds.
</shell_metadata>
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 4.0s)

_Thinking:_

The UI started but got killed because the command timed out. I need to run it in the background properly and keep it running. Let me use nohup or start it as a background process.

Actually, let me just tell the user to run the command directly in their terminal, as a background process.

Run this in your terminal:

```bash
cd ~/Projects/vehicles/fleet_loading && MLFLOW_ALLOW_FILE_STORE=true /tmp/ve_env2/bin/mlflow ui --backend-store-uri mlruns/ --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` in your browser. Press Ctrl+C to stop when done.

---

## User

vicbguti@vicbguti:~/Projects/vehicles$ cd ~/Projects/vehicles/fleet_loading && mlflow ui --backend-store-uri mlruns/
Command 'mlflow' not found, did you mean:
  command 'mflow' from deb mblaze (1.1-1)
Try: sudo apt install <deb name>
vicbguti@vicbguti:~/Projects/vehicles/fleet_loading$ cd ~/Projects/vehicles/fleet_loading && MLFLOW_ALLOW_FILE_STORE=true /tmp/ve_env2/bin/mlflow ui --backend-store-uri mlruns/ --host 127.0.0.1 --port 5000
Registry store URI not provided. Using backend store URI.
[MLflow] Security middleware enabled with default settings (localhost-only). To allow connections from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins.
2026/07/26 21:06:58 WARNING mlflow.server: MLflow job execution requirements not met (MLflow job backend requires a database backend store URI but got mlruns/). Server will start without job execution support. Errors will be surfaced at job invocation time.
2026/07/26 21:06:59 INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
2026/07/26 21:06:59 INFO:     Started parent process [177710]
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
/tmp/ve_env2/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:17: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
  from starlette.middleware.wsgi import WSGIResponder, build_environ
2026/07/26 21:07:01 INFO:     Started server process [177713]
2026/07/26 21:07:01 INFO:     Waiting for application startup.
2026/07/26 21:07:01 INFO:     Application startup complete.
2026/07/26 21:07:01 INFO:     Started server process [177714]
2026/07/26 21:07:01 INFO:     Waiting for application startup.
2026/07/26 21:07:01 INFO:     Application startup complete.
2026/07/26 21:07:01 INFO:     Started server process [177715]
2026/07/26 21:07:01 INFO:     Waiting for application startup.
2026/07/26 21:07:01 INFO:     Application startup complete.
2026/07/26 21:07:01 INFO:     Started server process [177712]
2026/07/26 21:07:01 INFO:     Waiting for application startup.
2026/07/26 21:07:01 INFO:     Application startup complete.
2026/07/26 21:07:05 INFO:     127.0.0.1:40564 - "GET / HTTP/1.1" 304 Not Modified
2026/07/26 21:07:05 INFO:     127.0.0.1:40564 - "GET /ajax-api/3.0/mlflow/server-info HTTP/1.1" 200 OK
2026/07/26 21:07:05 INFO:     127.0.0.1:40564 - "GET /ajax-api/2.0/mlflow/users/current HTTP/1.1" 404 Not Found
2026/07/26 21:07:05 INFO:     127.0.0.1:40580 - "GET /ajax-api/3.0/mlflow/assistant/config HTTP/1.1" 200 OK
WARNING:root:Malformed experiment '0'. Detailed error Yaml file 'mlruns/0/meta.yaml' does not exist.
Traceback (most recent call last):
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 386, in search_experiments
    exp = self._get_experiment(exp_id, view_type)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 487, in _get_experiment
    meta = FileStore._read_yaml(experiment_dir, FileStore.META_DATA_FILE_NAME)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1676, in _read_yaml
    return _read_helper(root, file_name, attempts_remaining=retries)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1669, in _read_helper
    result = read_yaml(root, file_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/utils/yaml_utils.py", line 104, in read_yaml
    raise MissingConfigException(f"Yaml file '{file_path}' does not exist.")
mlflow.exceptions.MissingConfigException: Yaml file 'mlruns/0/meta.yaml' does not exist.
2026/07/26 21:07:05 INFO:     127.0.0.1:40564 - "GET /ajax-api/2.0/mlflow/experiments/search?max_results=5&order_by=last_update_time+DESC HTTP/1.1" 200 OK
2026/07/26 21:07:06 INFO:     127.0.0.1:40568 - "GET /ajax-api/3.0/mlflow/ui-telemetry HTTP/1.1" 200 OK
WARNING:root:Malformed experiment '0'. Detailed error Yaml file 'mlruns/0/meta.yaml' does not exist.
Traceback (most recent call last):
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 386, in search_experiments
    exp = self._get_experiment(exp_id, view_type)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 487, in _get_experiment
    meta = FileStore._read_yaml(experiment_dir, FileStore.META_DATA_FILE_NAME)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1676, in _read_yaml
    return _read_helper(root, file_name, attempts_remaining=retries)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1669, in _read_helper
    result = read_yaml(root, file_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ve_env2/lib/python3.12/site-packages/mlflow/utils/yaml_utils.py", line 104, in read_yaml
    raise MissingConfigException(f"Yaml file '{file_path}' does not exist.")
mlflow.exceptions.MissingConfigException: Yaml file 'mlruns/0/meta.yaml' does not exist.
2026/07/26 21:07:07 INFO:     127.0.0.1:40568 - "GET /ajax-api/2.0/mlflow/experiments/search?max_results=25&order_by=last_update_time+DESC&filter=tags.%60mlflow.experiment.isGateway%60+IS+NULL HTTP/1.1" 200 OK 

---

## Assistant (Build · DeepSeek V4 Flash Free · 9.1s)

_Thinking:_

The mlruns directory is malformed — it's missing the `meta.yaml` file for experiment 0. This happens when the file store gets corrupted. The user has `mlflow.db` in the fleet_loading directory which is the SQLite store. Let me check what's in mlflow.db vs mlruns.

Actually, looking at this more carefully: the code is using the default MLflow tracking URI which is `file:./mlruns`. This writes to the `mlruns/` directory. But the `mlflow.db` was created separately somehow.

The simplest fix: change the tracking URI to use SQLite (the mlflow.db file) instead of the file store. But this means the existing runs in mlruns/ won't be visible unless we migrate.

Actually, let me just check: the mlflow.db might already contain the runs. Let me check.

The file store is corrupted. Let me switch to the SQLite backend that already exists:

**Tool: bash**

**Input:**
```json
{
  "command": "/tmp/ve_env2/bin/python -c \"\nimport mlflow\nmlflow.set_tracking_uri('sqlite:////home/vicbguti/Projects/vehicles/fleet_loading/mlflow.db')\nclient = mlflow.tracking.MlflowClient()\nfor exp in client.search_experiments():\n    print(f'Experiment: {exp.name} ({exp.experiment_id})')\n    for run in client.search_runs(exp.experiment_id, max_results=50):\n        print(f'  Run: {run.info.run_name} ({run.info.run_id})')\n        for k, v in run.data.metrics.items():\n            print(f'    {k}: {v}')\n\" 2>&1 | head -50",
  "workdir": "/home/vicbguti/Projects/vehicles/fleet_loading"
}
```

**Output:**
```
Experiment: Default (0)
  Run: xgboost_baseline (06440d30c6b44dacbab14ba4a8f535c4)
    xgb_val_accuracy: 0.958194197091125
    xgb_val_defer_f1: 0.029291406418264054
  Run: lightgbm_baseline (bf580bb98916455ea872adbed48fc93b)
    lgb_val_accuracy: 0.9576840308696942
    lgb_val_defer_f1: 0.0013134851138353765
  Run: attention_baseline (0c26dd7db8134654bc77f81acd024521)
    att_val_accuracy: 0.8121289700207777
    att_val_defer_f1: 0.6616689682383113
    att_cap_accuracy: 0.7582183140397745
    att_cap_defer_f1: 0.3103649635036496
  Run: xgboost_baseline (1438135c101940838448cf6b7db9be2a)
    xgb_val_accuracy: 0.958194197091125
    xgb_val_defer_f1: 0.029291406418264054
  Run: lightgbm_baseline (3334b93f4b3e4750a6431dc2e4c34a4b)
    lgb_val_accuracy: 0.9576840308696942
    lgb_val_defer_f1: 0.0013134851138353765
  Run: attention_baseline (f4e25dd14acd43bfb46eab5e9b0eb241)
    att_val_accuracy: 0.8125371029979223
    att_val_defer_f1: 0.665543031707574
  Run: xgboost_baseline (78e994e03c1a49c9a3870bd2f538223b)
    xgb_val_accuracy: 0.958194197091125
    xgb_val_defer_f1: 0.029291406418264054
  Run: lightgbm_baseline (28d630000d944fde9def4c0dc9d3c2c5)
    lgb_val_accuracy: 0.9576840308696942
    lgb_val_defer_f1: 0.0013134851138353765
  Run: xgboost_baseline (15e3faed4a594f19b301633f04b1f960)
    xgb_val_accuracy: 0.9582498515880084
    xgb_val_defer_f1: 0.031834803183480316
  Run: lightgbm_baseline (214a8e6d30264e278c3644015e18c217)
    lgb_val_accuracy: 0.957711858118136
    lgb_val_defer_f1: 0.003061447627378089
  Run: xgboost_baseline (91683d9fe184425398455f1673fcf8d3)
    xgb_val_accuracy: 0.9582684030869695
    xgb_val_defer_f1: 0.03309692671394799
  Run: lightgbm_baseline (135ee2f8d21e461da3c1004c8d5812b4)
    lgb_val_accuracy: 0.9576654793707332
    lgb_val_defer_f1: 0.0004380201489268506
  Run: xgboost_baseline (217c352e0c194608b6a076560e39d99f)
    xgb_val_accuracy: 0.9581292668447611
    xgb_val_defer_f1: 0.025053995680345574
  Run: xgboost_baseline (7a15e637b7364ac6ab97730642a6b379)
    xgb_val_accuracy: 0.9579344761056693
    xgb_val_defer_f1: 0.014772974147295241
  Run: xgboost_baseline (e2322d58844b436fa2d44eb148bed03b)
    xgb_val_accuracy: 0.9579344761056693
    xgb_val_defer_f1: 0.01520086862106406
  Run: lightgbm_baseline (ae7c4498f3444c6aaa14d855767e32b8)
    lgb_val_accuracy: 0.9576654793707332

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 6.3s)

_Thinking:_

The mlflow.db has all the runs! The issue is that the MLflow tracking is writing to the default file store (mlruns/), not to the SQLite db. But the SQLite db has somehow captured the runs from the file store — this is because the MLflow autologging or the tracking URI defaults might be saving to both?

Actually wait, the mlflow.db file might be getting created and updated by the Kedro MLflow hooks or by the MLflow runs themselves. Let me check the tracking URI that's being used.

Actually, looking at the code: `mlflow.start_run(run_name=...)` uses the default tracking URI, which is `file:./mlruns`. But somehow the mlflow.db also has the data. This is strange.

Let me just set MLFLOW_TRACKING_URI to use the sqlite backend for both training and UI. Let me check if the existing code already uses mlflow.db somehow.

Actually, let me look at the mlflow.db file more carefully. Maybe the user already configured MLflow to use sqlite and it's writing there via the environment variable.

Let me check the environment variables.

The SQLite db already has all runs. Let me just switch MLflow to use it permanently:

**Tool: bash**

**Input:**
```json
{
  "command": "echo $MLFLOW_TRACKING_URI; env | grep -i mlflow",
  "workdir": "/home/vicbguti/Projects/vehicles"
}
```

**Output:**
```


```

---

## Assistant (Build · DeepSeek V4 Flash Free · 10.7s)

_Thinking:_

No MLflow env vars set. The runs are being stored both in mlruns/ (file store, default) and mlflow.db (somehow). Actually wait — looking more carefully, maybe the `mlflow.db` is actually just a SQLite database that MLflow created because the Kedro project was initialized with MLflow support configured to use sqlite. But the file store is the default.

Let me just change the MLflow tracking URI to use sqlite and launch the UI pointing to it. But first, I should also make the training code use the same tracking URI so future runs go to the same place.

Actually, the simplest approach: tell the user the UI is broken because of file store corruption, but all data is in mlflow.db. Let me just start the UI with the sqlite backend.

**Tool: bash**

**Input:**
```json
{
  "command": "cd /home/vicbguti/Projects/vehicles/fleet_loading && MLFLOW_TRACKING_URI=sqlite:///mlflow.db /tmp/ve_env2/bin/mlflow ui 2>&1 &\nsleep 3\necho \"UI started. Go to http://127.0.0.1:5000\"",
  "timeout": 10000
}
```

**Output:**
```
Backend store URI not provided. Using sqlite:///mlflow.db
Registry store URI not provided. Using backend store URI.
[MLflow] Security middleware enabled with default settings (localhost-only). To allow connections from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins.
2026/07/26 21:07:51 ERROR:    [Errno 98] Address already in use
UI started. Go to http://127.0.0.1:5000

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 14.9s)

_Thinking:_

Port 5000 is already in use (from the previous mlflow ui that was started). Let me kill it first and restart.

Actually, I also need to set the tracking URI in the code so future training runs go to the same SQLite backend. Let me add `mlflow.set_tracking_uri` to the training functions.

But wait, there's also the issue with the `mlruns/` directory — the code uses the default file store. If I just add `mlflow.set_tracking_uri` in the code, future runs will write to the SQLite db, which already has all the data.

Let me kill the old process and restart.

**Tool: bash**

**Input:**
```json
{
  "command": "pkill -f \"mlflow ui\" 2>/dev/null; sleep 1\ncd /home/vicbguti/Projects/vehicles/fleet_loading && MLFLOW_TRACKING_URI=sqlite:///mlflow.db /tmp/ve_env2/bin/mlflow ui 2>&1 &\nsleep 2\necho \"UI at http://127.0.0.1:5000\"",
  "timeout": 10000
}
```

**Output:**
```
(no output)

<shell_metadata>
shell tool terminated command after exceeding timeout 10000 ms. If this command is expected to take longer and is not waiting for interactive input, retry with a larger timeout value in milliseconds.
</shell_metadata>
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 3.9s)

_Thinking:_

The command timed out but the UI might be running. Let me also need to set the MLflow tracking URI in the training code so future runs go to the SQLite database, not the corrupted file store.

Let me add `mlflow.set_tracking_uri` to the training functions in nodes.py.

Let me also fix the code to write to SQLite instead of the file store:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/vicbguti/Projects/vehicles/fleet_loading/src/fleet_loading/pipelines/training/nodes.py",
  "offset": 1,
  "limit": 12
}
```

**Output:**
```
<path>/home/vicbguti/Projects/vehicles/fleet_loading/src/fleet_loading/pipelines/training/nodes.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: import mlflow
4: import numpy as np
5: import pandas as pd
6: from sklearn.compose import ColumnTransformer
7: from sklearn.metrics import f1_score, accuracy_score
8: from sklearn.model_selection import GroupShuffleSplit
9: from sklearn.pipeline import Pipeline
10: from sklearn.preprocessing import OrdinalEncoder
11: 
12: NUMERIC_FEATURES = [

(Showing lines 1-12 of 312. Use offset=13 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 4.3s)

_Thinking:_

I'll add the tracking URI at the module level. But this is a Kedro project and the path is relative. Let me use an absolute path or a relative one that works.

**Tool: edit**

**Input:**
```json
{
  "filePath": "/home/vicbguti/Projects/vehicles/fleet_loading/src/fleet_loading/pipelines/training/nodes.py",
  "oldString": "from __future__ import annotations\n\nimport mlflow",
  "newString": "from __future__ import annotations\n\nimport os\n\nimport mlflow\n\nMLFLOW_DB = os.path.join(os.path.dirname(__file__), \"..\", \"..\", \"..\", \"..\", \"mlflow.db\")\nmlflow.set_tracking_uri(f\"sqlite:///{MLFLOW_DB}\")"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 5.6s)

_Thinking:_

Wait, this will set the tracking URI at import time, which is fine. But I also need to do the same in attention_model.py since it also uses mlflow. Let me check.

Actually, `mlflow.set_tracking_uri` sets a global process-level setting, so setting it once in nodes.py (which is imported before any mlflow calls) should be sufficient. The attention_model.py imports mlflow inside the function, but it will inherit the tracking URI that was set globally.

But wait, the attention_model.py has `import mlflow` inside the function body (local import). When mlflow is imported there, it uses the global tracking URI that was already set. So this should work.

Let me verify the path computation is correct by testing:

**Tool: bash**

**Input:**
```json
{
  "command": "/tmp/ve_env2/bin/python -c \"\nimport os\npath = os.path.join(os.path.dirname('/home/vicbguti/Projects/vehicles/fleet_loading/src/fleet_loading/pipelines/training/nodes.py'), '..', '..', '..', '..', 'mlflow.db')\nprint(os.path.abspath(path))\n\""
}
```

**Output:**
```
/home/vicbguti/Projects/vehicles/fleet_loading/mlflow.db

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 5.2s)

_Thinking:_

Good, the path resolves correctly. Now let me also fix the `mlflow.set_tracking_uri` to be set before any mlflow logging happens. The import happens at the top of the file, so by the time any training function is called, the tracking URI will already be set.

But I also need to make sure this works with the existing `mlflow.log_param` calls. Let me also check the pipeline's MLflow setup — Kedro might have its own MLflow hooks.

Actually, Kedro 1.5.0 doesn't have built-in MLflow hooks. The mlflow usage is manual in the training functions. So setting the tracking URI at import time should work.

Let me now commit this fix and tell the user to restart the UI.

Now the training writes to SQLite. Run the UI in your terminal:

```bash
cd ~/Projects/vehicles/fleet_loading && MLFLOW_TRACKING_URI=sqlite:///mlflow.db /tmp/ve_env2/bin/mlflow ui
```

Then `http://127.0.0.1:5000` should show all runs including the latest attention results.

---

