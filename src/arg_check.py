# Created by ay27 at 16/11/9
import functools


def arg_check(*args):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*inner_args, **kw):
            for ii in range(len(args)):
                if args[ii] is None:
                    continue
                if isinstance(args[ii], list) or isinstance(args[ii], tuple):
                    if not type(inner_args[ii]) in args[ii]:
                        raise ValueError('args check error')
                else:
                    if not type(inner_args[ii]) == args[ii]:
                        raise ValueError('args check error')
            return func(*inner_args, **kw)

        return wrapper

    return decorator
