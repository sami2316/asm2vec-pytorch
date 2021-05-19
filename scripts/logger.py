'''
:mod:`logger` -- Logging helper functions
=========================================
.. module: logger
   :platform: Unix, Windows
   :synopsis: Logging helper functions


About
-----
This module uses Python's :mod:`logging` module to create a simple hierarchy of
logger objects used by various classes of REveal.

A class, whose methods need to write to the log, should do something like the
following to initialize its own private logger instance.

.. code-block:: python

   class LogProducer(object):
       def __init__(self):
           super(LogProducer, self).__init__()
           self.logger = logger.get_logger(self.__class__.__name__)

Methods of this class can now use ``self.logger`` to write messages to REveal's
log file.


Definitions
-----------
'''

__all__ = ['get_root_logger', 'set_root_logger_level', 'get_logger', 'shutdown']


import logging

LOG_FILE = None

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(msg)s',
    datefmt='%Y-%m-%d %H:%M:%S')


_ROOT_LOGGER_NAME = 'Asm2Vec'

_ROOT_LOGGER = None


def get_root_logger():
    '''
    Get REveal's root logger object instance.

    :returns: REveal's root logger
    :rtype: logging.Logger
    '''

    global _ROOT_LOGGER

    if _ROOT_LOGGER is None:
        _ROOT_LOGGER = logging.getLogger(_ROOT_LOGGER_NAME)
        if LOG_FILE is not None:
            _ROOT_LOGGER.addHandler(logging.FileHandler(LOG_FILE))
        _ROOT_LOGGER.setLevel(logging.DEBUG)

    return _ROOT_LOGGER


def set_root_logger_level(level):
    '''
    Set root logger's logging level. See [01] for more information.

    [01] `<https://docs.python.org/2/library/logging.html#logging-levels>`_

    :param int level: The desired log level constant
    '''
    get_root_logger().setLevel(level)


def get_logger(name):
    '''
    Create a new logger identified by a given name.

    :param str name: Name of the new logger
    :returns: The new logger object instance
    :rtype: logging.Logger
    '''
    return logging.getLogger('%s.%s' % (get_root_logger().name, name))


def shutdown():
    '''
    Shutdown all loggers.
    '''
    logging.shutdown()

