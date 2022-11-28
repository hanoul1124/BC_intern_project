import logging
import logging.config

config = {
    "version": 1.0,
    "formatters": {
        "simple": {
            "format":"%(asctime)s [%(name)s] %(message)s",
        },
        "complex": {
            "format": "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "crawler.log",
            "formatter": "complex",
            "level": "ERROR",
        },
    },
    "root": {"handlers": ["console", "file"], "level": "WARNING"},
    "loggers": {"parent": {"level": "INFO"}, "parent.child": {"level": "DEBUG"},},
}

logging.config.dictConfig(config)
