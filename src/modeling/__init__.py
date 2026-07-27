"""Modelado supervisado del asignador vehículo-camión (Tarea #4, Juan Francisco).

Este paquete es **aditivo**: no modifica `src/loading/labeler.py` ni
`src/loading/scenarios.py`, que son el maestro exacto y el generador de episodios
compartidos por todo el equipo. Todo lo que hay aquí consume los dos parquet que
`scripts/build_scenarios.py` ya produce.

Módulos
-------
canonicalization  Reordena la flota por capacidad y remapea las etiquetas de camión.
dataset           Join episodios ⋈ vehículos y particiones que respetan el episodio.
features          Tensores por par (vehículo, camión) + contexto del manifiesto.
mlp_classifier    El MLP compartido en Keras 3.
capacity_decoder  Decodificación factible (nunca excede capacidad).
metrics           Métricas de dominio contra el maestro exacto.
"""
