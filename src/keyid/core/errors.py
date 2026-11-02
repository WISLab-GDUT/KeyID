from __future__ import annotations

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_PARTIAL = 2


class KeyIDError(Exception):
    pass


class ConfigError(KeyIDError):
    pass


class DatasetError(KeyIDError):
    pass


class GraphError(KeyIDError):
    pass


class RegistryError(KeyIDError):
    pass


class StageError(KeyIDError):
    pass


class ArtifactError(KeyIDError):
    pass
