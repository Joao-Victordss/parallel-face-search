"""Pontos de entrada de linha de comando.

Cada modulo deste subpacote expoe uma funcao ``main()`` ligada a um comando
instalavel (ver ``[project.scripts]`` no ``pyproject.toml``):

- ``check_environment``  -> face-search-check
- ``sync_gallery``       -> face-search-sync
- ``webcam_search``      -> face-search-webcam
- ``build_local_gallery``-> face-search-build-gallery
- ``match_images``       -> face-search-match
- ``evaluate_accuracy``  -> face-search-evaluate

Os modulos de CLI sao finos de proposito: cuidam apenas de argumentos e
validacao, delegando a logica de verdade aos subpacotes do ``face_search``.
"""
