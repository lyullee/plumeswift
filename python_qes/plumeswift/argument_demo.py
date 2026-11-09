"""Python counterpart to the public scratch/argparser.cpp utility."""
_SHORT = {'?': 'help', 'v': 'verbose', 'q': 'quicproj'}
_HELP = """Allowed options:
  -? [ --help ]         help/usage information
  -v [ --verbose ]      turn on verbose output
  -q [ --quicproj ] arg Specifies the QUIC Proj file

"""


def argument_demo(arguments):
    """Return the scratch utility output or raise a command-line error."""
    options = []
    project = ''
    pending = iter(arguments)
    for token in pending:
        if token == '--':
            break
        if token.startswith('--'):
            spelling, equal, value = token[2:].partition('=')
            if not spelling:
                # Boost treats --=value as positional text.
                if not value:
                    raise ValueError('the argument for option should follow immediately after the equal sign')
                continue
            name = next((name for name in _SHORT.values() if name.startswith(spelling)), None)
            if name is None:
                raise ValueError(f"unrecognised option '{token}'")
            if equal and not value:
                raise ValueError(f"the argument for option '--{name}' should follow immediately after the equal sign")
            if equal and name != 'quicproj':
                raise ValueError(f"option '--{name}' does not take any arguments")
            entries = ((name, value if equal else None),)
        elif token.startswith('-') and len(token) > 1:
            entries = []
            for index, letter in enumerate(token[1:], 2):
                name = _SHORT.get(letter)
                if name is None:
                    raise ValueError(f"unrecognised option '{token}'")
                attached = token[index:] if name == 'quicproj' else ''
                entries.append((name, attached or None))
                if name == 'quicproj':
                    break
        else:
            continue
        for name, value in entries:
            options.append(name)
            if name == 'quicproj':
                if value is None:
                    value = next(pending, None)
                    if value is None or value in ('-?', '-v', '-q'):
                        raise ValueError("the required argument for option '--quicproj' is missing")
                project = value
    seen = set()
    for name in options:
        if name in seen:
            raise ValueError(f"option '--{name}' cannot be specified more than once")
        seen.add(name)
    if 'help' in seen:
        return _HELP
    return f'Verbose Output: ON\nquicproj set to {project}\n' if 'verbose' in seen else ''
