"""Pure presentation of an already qualified nominal net mass, never a sensor."""
import math

SEGMENTS = {
    '0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg',
    '5': 'acdfg', '6': 'acdefg', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg', '-': 'g',
}


def readout(mass_g=None, *, stable_s=0.):
    """A 00.0 g display; unknown/unloaded/unsettled states show three dashes."""
    valid = (type(mass_g) in (int, float) and math.isfinite(mass_g) and 0 <= mass_g <= 99.9)
    text = f'{mass_g:04.1f}' if valid else '---'
    digits = text.replace('.', '')
    segments = {f'digit_{i}_{segment}': segment in SEGMENTS[character]
                for i, character in enumerate(digits) for segment in 'abcdefg'}
    segments.update(decimal=valid, stable=valid)
    return {'value_g': float(mass_g) if valid else None, 'text': text, 'unit': 'g',
        'valid': valid, 'stable_s': float(stable_s), 'segments': segments,
        'basis': 'nominal in-cup granule inventory; empty boat tared; not force sensing'}
