is_simple_core = False

if is_simple_core:
    from deepy.core_simple import Variable
    from deepy.core_simple import Function
    from deepy.core_simple import using_config
    from deepy.core_simple import no_grad
    from deepy.core_simple import as_array
    from deepy.core_simple import as_variable
    from deepy.core_simple import setup_variable

else:
    from deepy.core import Variable
    from deepy.core import Parameter
    from deepy.core import Function
    from deepy.core import using_config
    from deepy.core import no_grad
    from deepy.core import as_array
    from deepy.core import as_variable
    from deepy.core import setup_variable
    from deepy.core import Config
    from deepy.layers import Layer
    from deepy.models import Model

setup_variable()
